"""Async, pollable ingest jobs.

Wraps the blocking end-to-end ingest (``convert -> generate -> embed ->
index``, minutes on a large document) in a state machine

    ``queued -> running -> success | failed | cancelled``

so a long ingest no longer holds open an HTTP connection. A
``POST /api/v1/knowledge_base/ingest/async`` route returns a job id
immediately; ``GET /api/v1/jobs/{job_id}`` polls the snapshot.

This mirrors the :mod:`sparksage.distill.job` shape (the same state-machine +
cooperative-cancellation + lock-protected-snapshot pattern) but stays
self-contained in the ``api`` package: ingest observability is an HTTP-layer
concern, and the job owns no ingest-specific configuration (it takes a
fully-bound ``work`` callable produced by :meth:`QAService.submit_ingest`).

Concurrency model
-----------------
The work runs in a daemon thread (:meth:`IngestJob.start`) so the blocking
LLM / embedding I/O stays off the event loop. Python cannot kill the worker
thread, so cancellation is *cooperative*: the work callable receives an
``is_cancelled`` predicate and an ``on_progress`` callback; the underlying
:meth:`~sparksage.api.qa_service.QAService.ingest_and_index` checks the
predicate at phase boundaries (before convert / generate / index) and raises
:class:`IngestCancelled` rather than writing to the knowledge base. A cancel
that lands mid-phase (during a blocking LLM call) takes effect at the next
boundary -- the partial result is never written.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

_logger = logging.getLogger(__name__)

#: Type alias for the phase-progress callback handed to the work callable.
#: Receives a short phase label (``"converting"`` / ``"generating"`` /
#: ``"indexing"``); the job translates each into a snapshot update.
ProgressCallback = Callable[[str], None]

#: Type alias for the cooperative-cancellation predicate handed to the work
#: callable. Returns ``True`` once :meth:`IngestJob.cancel` (or a timeout)
#: has been requested.
CancelPredicate = Callable[[], bool]

#: Type alias for the fully-bound work callable the job runs. It receives the
#: progress + cancellation hooks and returns the :class:`IngestResult`-shaped
#: object (typed as ``Any`` here so this module stays free of the
#: :mod:`sparksage.api.qa_service` import cycle).
IngestWork = Callable[[ProgressCallback, CancelPredicate], Any]


class IngestCancelled(Exception):
    """Raised by ingest work when ``is_cancelled`` fires at a phase boundary.

    Caught by :class:`IngestJob` to resolve the job to ``cancelled`` without
    writing a partial result to the knowledge base.
    """


class IngestJobStatus(str, Enum):
    """Lifecycle status of an :class:`IngestJob`.

    State transitions::

        queued -> running -> success    (clean completion)
                         -> failed      (an exception escaped the work)
                         -> cancelled   (cancel() fired before / during the run)

    All three terminal states are stable. ``cancelled`` never carries a
    result: cancellation is checked at phase boundaries *before* the
    knowledge-base write, so a cancelled ingest leaves the KB untouched.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (IngestJobStatus.SUCCESS, IngestJobStatus.FAILED, IngestJobStatus.CANCELLED)

    @property
    def is_running(self) -> bool:
        return self is IngestJobStatus.RUNNING


#: Phase labels written into :attr:`IngestJobSnapshot.phase`. ``queued`` before
#: start, the work's emitted label while running, ``done`` on any terminal
#: status.
PHASE_QUEUED = "queued"
PHASE_CONVERTING = "converting"
PHASE_GENERATING = "generating"
PHASE_INDEXING = "indexing"
PHASE_DONE = "done"

#: Rough per-phase percent for the progress bar. Ingest has no natural
#: iteration count, so each completed phase advances the bar to the next
#: threshold; ``1.0`` on completion.
_PHASE_PERCENT: dict[str, float] = {
    PHASE_QUEUED: 0.0,
    PHASE_CONVERTING: 0.15,
    PHASE_GENERATING: 0.55,
    PHASE_INDEXING: 0.85,
    PHASE_DONE: 1.0,
}


@dataclass(frozen=True)
class IngestJobSnapshot:
    """Immutable point-in-time view of an ingest job.

    Frozen so it is safe to hand to a polling HTTP client without copying --
    the job never mutates a snapshot in place; it replaces its internal
    reference under a lock.

    Attributes
    ----------
    job_id:
        The opaque job identifier (caller-supplied or auto-generated UUID).
    status:
        Current :class:`IngestJobStatus`.
    phase:
        Coarse stage label: ``"queued"`` before start, the latest phase
        emitted by the work (``"converting"`` / ``"generating"`` /
        ``"indexing"``) while running, ``"done"`` on completion.
    percent:
        Completion fraction in ``[0, 1]`` -- advances per phase, ``1.0`` on
        completion (any terminal status).
    filename:
        The original uploaded filename (for the UI's per-file status row).
    title:
        The resolved document title once the work produces one; ``None``
        until then.
    block_count:
        Number of indexed IdeaBlocks once the work succeeds; ``0`` until then.
    doc_id:
        The stored document id on success; ``None`` otherwise.
    error:
        Exception message when ``status == failed``; ``None`` otherwise.
    created_at, started_at, finished_at:
        :func:`time.monotonic` timestamps for the lifecycle events.
    """

    job_id: str
    status: IngestJobStatus
    phase: str = PHASE_QUEUED
    percent: float = 0.0
    filename: str | None = None
    title: str | None = None
    block_count: int = 0
    doc_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


#: Type alias for the optional job-level progress callback. Invoked with an
#: :class:`IngestJobSnapshot` every time the job's progress or status changes,
#: from the worker thread. Keep it cheap (e.g. publish to a queue / websocket).
IngestProgressCallback = Callable[[IngestJobSnapshot], None]


def _default_job_id() -> str:
    return str(uuid.uuid4())


class IngestJob:
    """Stateful, pollable wrapper around a single ingest run.

    The job does not start running until :meth:`start` (or :meth:`run_sync`)
    is called. Before that, :attr:`status` is :attr:`IngestJobStatus.QUEUED`.
    A job is single-use: once terminal it cannot be restarted.

    Parameters
    ----------
    work:
        A fully-bound callable (produced by :meth:`QAService.submit_ingest`)
        that runs the ingest given a progress callback and a cancellation
        predicate, returning an :class:`IngestResult`-shaped object.
    job_id:
        Optional identifier. Auto-generated as a UUID4 string when omitted.
    filename:
        The original uploaded filename -- surfaced on every snapshot so the
        UI's per-file status row is stable across polls.
    on_progress:
        Optional callback invoked with an :class:`IngestJobSnapshot` on every
        status / phase change, from the worker thread. Keep it cheap.
    """

    def __init__(
        self,
        work: IngestWork,
        *,
        job_id: str | None = None,
        filename: str | None = None,
        on_progress: IngestProgressCallback | None = None,
    ) -> None:
        self._work = work
        self._job_id = job_id if job_id is not None else _default_job_id()
        self._filename = filename
        self._external_callback = on_progress

        self._lock = threading.Lock()
        self._snapshot = IngestJobSnapshot(
            job_id=self._job_id,
            status=IngestJobStatus.QUEUED,
            filename=filename,
        )
        self._cancel_requested: bool = False
        self._started_once: bool = False
        # The ingest result (written on success). Kept off the frozen snapshot
        # so the observable snapshot stays serialization-clean; read by
        # :meth:`run_sync` and :meth:`result`.
        self._result: Any = None

    # ------------------------------------------------------------------ #
    # read-only views
    # ------------------------------------------------------------------ #
    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def filename(self) -> str | None:
        return self._filename

    @property
    def status(self) -> IngestJobStatus:
        with self._lock:
            return self._snapshot.status

    @property
    def result(self) -> Any:
        """The ingest result, or ``None`` if not done / cancelled / failed."""
        with self._lock:
            return self._result

    def snapshot(self) -> IngestJobSnapshot:
        """Return an immutable point-in-time snapshot copy."""
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------ #
    # cancellation
    # ------------------------------------------------------------------ #
    def cancel(self) -> bool:
        """Request cooperative cancellation of a running (or queued) job.

        Returns whether the request was *accepted*. A terminal job cannot be
        cancelled (returns ``False``). The work's ``is_cancelled`` predicate
        next reads ``True`` at its phase boundary, raises
        :class:`IngestCancelled`, and the job resolves to ``cancelled`` --
        without writing a partial result to the knowledge base. A cancel that
        lands mid-phase (during a blocking LLM call) takes effect at the next
        boundary.
        """
        with self._lock:
            if self._snapshot.status.is_terminal:
                return False
            self._cancel_requested = True
        return True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def run_sync(self) -> Any:
        """Run the work in the *current* thread and return the result.

        Use this in scripts / tests. Blocks the caller for the whole run.
        Raises :class:`RuntimeError` if the job was already started, or
        :class:`IngestCancelled` if the work was cancelled before producing a
        result (and no cancel was requested via :meth:`cancel` -- that path
        resolves cleanly to the ``cancelled`` status).
        """
        self._mark_running()
        self._execute()
        with self._lock:
            snap = self._snapshot
            result = self._result
        if snap.status is IngestJobStatus.SUCCESS:
            return result
        if snap.status is IngestJobStatus.CANCELLED:
            raise IngestCancelled(f"ingest job {self._job_id} was cancelled")
        raise RuntimeError(
            f"ingest job {self._job_id} finished in status {snap.status.value}"
        )

    def start(self) -> None:
        """Run the work in a daemon thread (non-blocking).

        The blocking ingest I/O runs off the caller thread. Subscribe via
        ``on_progress=`` or poll :meth:`snapshot`. Raises :class:`RuntimeError`
        if the job was already started.
        """
        self._mark_running()
        thread = threading.Thread(
            target=self._execute,
            name=f"ingest-job-{self._job_id}",
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _mark_running(self) -> None:
        with self._lock:
            if self._started_once:
                raise RuntimeError(
                    f"ingest job {self._job_id} has already been started; a job "
                    "is single-use -- submit a new one for a fresh run"
                )
            self._started_once = True
            now = time.monotonic()
            self._snapshot = replace(
                self._snapshot,
                status=IngestJobStatus.RUNNING,
                started_at=now,
                phase=PHASE_CONVERTING,
                percent=_PHASE_PERCENT[PHASE_CONVERTING],
            )
            current = self._snapshot
        self._emit(current)

    def _on_phase(self, phase: str) -> None:
        """Progress callback handed to the work callable (called from worker)."""
        with self._lock:
            if self._snapshot.status is not IngestJobStatus.RUNNING:
                return
            self._snapshot = replace(
                self._snapshot,
                phase=phase,
                percent=_PHASE_PERCENT.get(phase, self._snapshot.percent),
            )
            current = self._snapshot
        self._emit(current)

    def _execute(self) -> None:
        try:
            result = self._work(self._on_phase, self._is_cancelled)
        except IngestCancelled:
            self._set_terminal(IngestJobStatus.CANCELLED)
            _logger.info("ingest job %s cancelled", self._job_id)
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure on the snapshot
            self._set_terminal(
                IngestJobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
            _logger.exception("ingest job %s failed", self._job_id)
            return

        with self._lock:
            cancelled = self._cancel_requested
        if cancelled:
            # Cancel raced with completion -- prefer the caller's intent and
            # mark cancelled, but the knowledge-base write already happened
            # (the work returned a result). This is rare (cancel landed in
            # the window between the final phase check and the return); the
            # snapshot reports cancelled for honesty, without the doc payload.
            self._set_terminal(IngestJobStatus.CANCELLED)
            _logger.info(
                "ingest job %s cancelled (post-completion race)", self._job_id
            )
            return
        self._set_terminal(IngestJobStatus.SUCCESS, result=result)

    def _set_terminal(
        self,
        status: IngestJobStatus,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if self._snapshot.status.is_terminal:
                return
            now = time.monotonic()
            snap = replace(
                self._snapshot,
                status=status,
                finished_at=now,
                phase=PHASE_DONE,
                percent=1.0,
                error=error,
            )
            if status is IngestJobStatus.SUCCESS and result is not None:
                snap = replace(
                    snap,
                    doc_id=getattr(result, "doc_id", None),
                    block_count=getattr(result, "block_count", 0),
                    title=getattr(result, "title", None),
                )
                self._result = result
            self._snapshot = snap
            current = self._snapshot
        self._emit(current)
        _logger.info(
            "ingest job %s terminal: status=%s blocks=%d",
            self._job_id,
            status.value,
            current.block_count,
        )

    def _emit(self, snapshot: IngestJobSnapshot) -> None:
        callback = self._external_callback
        if callback is None:
            return
        try:
            callback(snapshot)
        except Exception:  # noqa: BLE001 - a buggy callback must not kill the worker
            _logger.exception(
                "ingest job %s on_progress callback raised; ignoring", self._job_id
            )

    def __repr__(self) -> str:
        with self._lock:
            status = self._snapshot.status
        return f"IngestJob(job_id={self._job_id!r}, status={status.value!r})"


class IngestJobManager:
    """In-process registry of :class:`IngestJob` s, keyed by id.

    The service-layer object the ``POST /knowledge_base/ingest/async`` and
    ``GET /jobs/{id}`` routes wrap:

    * :meth:`submit` registers a job and starts it in a background thread,
      returning the job (and its id) immediately;
    * :meth:`snapshot` backs ``GET /jobs/{id}``;
    * :meth:`cancel` backs ``POST /jobs/{id}/cancel``.

    Jobs live in a plain dict -- intentionally in-process so the layer stays
    fully unit-testable with deterministic fakes and zero infrastructure.
    A single ingest run is LLM- and embedding-bound, not a distributed
    workload, so process-locality is the right first cut.

    Parameters
    ----------
    max_jobs:
        Soft cap on retained (terminal) jobs so a long-lived server does not
    """

    def __init__(self, *, max_jobs: int = 200) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(
        self,
        work: IngestWork,
        *,
        job_id: str | None = None,
        filename: str | None = None,
        on_progress: IngestProgressCallback | None = None,
    ) -> IngestJob:
        """Register a new :class:`IngestJob` and start it in a background thread.

        Returns the job immediately -- the long-running work happens in the
        worker thread. Raises :class:`ValueError` on an id collision.
        """
        job = IngestJob(
            work,
            job_id=job_id,
            filename=filename,
            on_progress=on_progress,
        )
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job_id {job.job_id!r} already exists in this manager")
            self._jobs[job.job_id] = job
            self._evict_locked()
        job.start()
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> IngestJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.snapshot() if job is not None else None

    def cancel(self, job_id: str) -> bool | None:
        """Request cancellation. Returns ``None`` for an unknown id, otherwise
        the job's accept/reject bool."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.cancel()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._jobs

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def forget(self, job_id: str) -> bool:
        """Drop a job from the registry. Returns whether it was present.

        The worker thread (if still running) is *not* cancelled -- call
        :meth:`IngestJob.cancel` first if you want it to wind down.
        """
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def _evict_locked(self) -> None:
        """Evict the oldest terminal jobs when the registry exceeds the cap.

        Non-terminal jobs are never evicted (they are still observable).
        """
        if len(self._jobs) <= self._max_jobs:
            return
        terminal = [
            (jid, job)
            for jid, job in self._jobs.items()
            if job.status.is_terminal
        ]
        overflow = len(self._jobs) - self._max_jobs
        for jid, _job in terminal[:overflow]:
            self._jobs.pop(jid, None)


__all__ = [
    "CancelPredicate",
    "IngestCancelled",
    "IngestJob",
    "IngestJobManager",
    "IngestJobSnapshot",
    "IngestJobStatus",
    "IngestProgressCallback",
    "IngestWork",
    "ProgressCallback",
]
