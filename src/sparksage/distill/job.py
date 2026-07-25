"""Async, pollable Distill jobs: wrap :class:`DistillPipeline` in a state machine.

For large corpora the Distill pipeline is a long-running, LLM- and
embedding-heavy job -- minutes on 10k blocks, hours on a million. A
synchronous ``pipeline.run(blocks)`` call blocks the caller for the whole
duration with no visibility. This module wraps the pipeline in a
:class:`DistillJob` state machine

    ``queued -> running -> success | failed | timeout | cancelled``

that emits :class:`JobSnapshot` progress (percent / phase / iteration /
threshold / block_count / candidate_pairs, mirroring the
``{percent, phase, details}`` shape a polling REST client expects), supports
cooperative cancellation and timeout, and can be driven synchronously
(:meth:`DistillJob.run_sync`) or asynchronously (:meth:`DistillJob.start`).

The job does *not* own pipeline configuration -- it takes a fully-configured
:class:`~sparksage.distill.DistillPipeline` (whose ``candidate_reducer`` /
clustering backend / threshold schedule were chosen up front). The job only
owns run state: status, timing, progress, error, and the eventual
:class:`~sparksage.distill.DistillResult`.

:class:`JobManager` is the service-layer object a future ``/api/v1/distill``
route will wrap: ``submit()`` returns a job id immediately, ``snapshot(id)``
backs ``GET /jobs/{id}``, and ``wait_for(id)`` backs a long-poll or websocket
flush. It keeps the indexed jobs in a plain dict -- intentionally in-process so
the whole layer stays unit-testable with the deterministic fakes and zero
infrastructure. Process-locality is the right first cut: a single Distill run
is CPU- and LLM-bound, not a distributed workload.

Concurrency model
-----------------
``DistillPipeline.run`` is blocking (CPU + LLM I/O). The async
:meth:`DistillJob.start` runs it in a worker thread via
:func:`asyncio.to_thread` and awaits it; the pipeline's ``on_progress``
callback (invoked from the worker thread) updates a lock-protected snapshot
field. Python cannot kill the worker thread, so cancellation is *cooperative*:
the pipeline polls an ``is_cancelled`` predicate at iteration boundaries and
returns the partial result; ``cancel()`` / timeout both flip that predicate so
the worker exits promptly without wasting further LLM/embedding calls. The
partial result is preserved on the snapshot for inspection.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum

from sparksage.distill.pipeline import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_START_THRESHOLD,
    DistillPipeline,
    DistillProgress,
    DistillResult,
)
from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)

#: Soft default for :meth:`DistillJob.start` ``timeout`` when the caller passes
#: ``None``: wait forever. Exposed so callers can resolve it from one place.
DEFAULT_JOB_TIMEOUT: float | None = None

#: Internal progress emitted before the first pipeline callback lands (and for
#: the empty-input fast path). Kept module-private: callers read
#: :attr:`JobSnapshot.progress`, not this constant.
_INITIAL_PERCENT: float = 0.0


class JobStatus(str, Enum):
    """Lifecycle status of a :class:`DistillJob`.

    State transitions::

        queued -> running -> success   (clean completion)
                          -> failed    (an exception escaped the pipeline)
                          -> timeout   (the async deadline elapsed)
                          -> cancelled (cancel() was called before completion)

    All four terminal states are stable: a terminal job never transitions
    again. ``timeout`` and ``cancelled`` retain any partial result the
    pipeline produced before stopping -- inspected via
    :attr:`JobSnapshot.result`.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """``True`` for ``success`` / ``failed`` / ``timeout`` / ``cancelled``."""
        return self in (
            JobStatus.SUCCESS,
            JobStatus.FAILED,
            JobStatus.TIMEOUT,
            JobStatus.CANCELLED,
        )

    @property
    def is_running(self) -> bool:
        """``True`` for the active ``running`` state."""
        return self is JobStatus.RUNNING


@dataclass(frozen=True)
class JobProgress:
    """Live progress view of a running (or finished) job.

    Mirrors the ``{percent, phase, details}`` shape a polling REST client
    expects: ``percent`` for the progress bar, ``phase`` for the human label,
    and the per-iteration ``details`` (iteration / threshold / block_count /
    candidate_pairs / ...) for the diagnostics panel.

    Attributes
    ----------
    percent:
        Completion fraction in ``[0, 1]`` -- ``iteration / max_iterations``,
        snapping to ``1.0`` once the run is done. ``0.0`` before iteration 1.
    phase:
        Coarse stage label: ``"queued"`` before start, ``"running"`` while
        iterations execute, ``"done"`` on completion (any terminal status).
    iteration:
        1-based index of the latest iteration that has *started* (``0`` before
        iteration 1, ``max_iterations`` at most).
    max_iterations:
        The hard cap on rounds (from the pipeline config), so callers can render
        ``iteration / max_iterations``.
    threshold:
        Similarity threshold in effect for the latest iteration.
    active_blocks:
        Number of blocks in the active set at the latest emission.
    candidate_pairs, merge_clusters, blocks_merged, canonical_emitted:
        Latest iteration's counters (``0`` until iteration 1 completes).
    """

    percent: float
    phase: str
    iteration: int
    max_iterations: int
    threshold: float | None
    active_blocks: int
    candidate_pairs: int = 0
    merge_clusters: int = 0
    blocks_merged: int = 0
    canonical_emitted: int = 0


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable point-in-time view of a job, returned by :meth:`DistillJob.snapshot`.

    Frozen so it is safe to hand to a polling client without copying -- the job
    never mutates a snapshot in place; it replaces its internal reference under
    a lock.

    Attributes
    ----------
    job_id:
        The opaque job identifier (caller-supplied or auto-generated UUID).
    status:
        Current :class:`JobStatus`.
    input_blocks:
        Size of the original block list handed to the job.
    created_at, started_at, finished_at:
        :func:`time.monotonic` timestamps for the lifecycle events.
        ``None`` until the corresponding event happens. Prefer these over wall
        clock for elapsed-time computations -- they are monotonic and immune to
        system clock skew.
    progress:
        Latest :class:`JobProgress`.
    error:
        Exception message when ``status == failed``; ``None`` otherwise.
    result:
        The :class:`~sparksage.distill.DistillResult` when
        ``status == success`` (and any partial result for ``timeout`` /
        ``cancelled``); ``None`` while running or on failure.
    """

    job_id: str
    status: JobStatus
    input_blocks: int
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    progress: JobProgress = field(
        default_factory=lambda: JobProgress(
            percent=_INITIAL_PERCENT,
            phase=JobStatus.QUEUED.value,
            iteration=0,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            threshold=DEFAULT_START_THRESHOLD,
            active_blocks=0,
        )
    )
    error: str | None = None
    result: DistillResult | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether :attr:`status` is terminal (the job will not change further)."""
        return self.status.is_terminal

    @property
    def duration(self) -> float | None:
        """Elapsed seconds from start to finish, or ``None`` if not yet started.

        For a running job: seconds since :attr:`started_at`. For a finished
        job: ``finished_at - started_at``. ``None`` before the job starts.
        """
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


#: Type alias for the optional job-level progress callback. Invoked with a
#: :class:`JobSnapshot` every time the job's progress or status changes, from
#: the worker thread. Keep it cheap (e.g. publish to a queue / websocket).
JobProgressCallback = Callable[[JobSnapshot], None]


def _default_job_id() -> str:
    return str(uuid.uuid4())


def _progress_from_pipeline(
    progress: DistillProgress,
) -> JobProgress:
    """Translate a :class:`DistillProgress` event into the public :class:`JobProgress`."""
    snap = progress.snapshot
    if snap is not None:
        return JobProgress(
            percent=progress.percent,
            phase=progress.phase,
            iteration=snap.iteration,
            max_iterations=progress.max_iterations,
            threshold=snap.threshold,
            active_blocks=snap.active_blocks,
            candidate_pairs=snap.candidate_pairs,
            merge_clusters=snap.merge_clusters,
            blocks_merged=snap.blocks_merged,
            canonical_emitted=snap.canonical_emitted,
        )
    return JobProgress(
        percent=progress.percent,
        phase=progress.phase,
        iteration=progress.iteration,
        max_iterations=progress.max_iterations,
        threshold=progress.threshold,
        active_blocks=progress.active_blocks,
    )


class DistillJob:
    """Stateful, pollable wrapper around a single :class:`DistillPipeline` run.

    The job does not start running until :meth:`run_sync` or :meth:`start` is
    called. Before that, :attr:`status` is :attr:`JobStatus.QUEUED`. Once
    started, the pipeline runs (in the calling thread for :meth:`run_sync`, in
    a worker thread for :meth:`start`) and the job transitions to
    :attr:`JobStatus.RUNNING` and then to a terminal state.

    A job is single-use: once it reaches a terminal state it cannot be restarted.
    Call :meth:`JobManager.submit` (or construct a new :class:`DistillJob`) for
    a fresh run.

    Parameters
    ----------
    blocks:
        The :class:`~sparksage.schema.IdeaBlock` corpus to de-duplicate. Stored
        by reference; the pipeline does not mutate it.
    pipeline:
        A fully-configured :class:`DistillPipeline` (threshold schedule,
        clustering backend, candidate reducer, etc.). The job only adds run
        state around it.
    job_id:
        Optional identifier. Auto-generated as a UUID4 string when omitted.
        Useful for correlation with an upstream request id.
    on_progress:
        Optional callback invoked with a :class:`JobSnapshot` on every status
        or progress change, from the worker thread. Keep it cheap.

    Examples
    --------
    Synchronous (script / REPL)::

        job = DistillJob(blocks, pipeline)
        result = job.run_sync()
        print(job.snapshot().status)  # JobStatus.SUCCESS

    Asynchronous (FastAPI / any asyncio app)::

        job = DistillJob(blocks, pipeline, job_id="run-42")
        result = await job.start(timeout=3600.0)
        # ...elsewhere, polling:
        snap = job.snapshot()
        print(snap.progress.percent, snap.progress.iteration)
    """

    def __init__(
        self,
        blocks: list[IdeaBlock],
        pipeline: DistillPipeline,
        *,
        job_id: str | None = None,
        on_progress: JobProgressCallback | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._blocks: list[IdeaBlock] = list(blocks)
        self._job_id = job_id if job_id is not None else _default_job_id()
        self._external_callback = on_progress

        self._lock = threading.Lock()
        self._snapshot: JobSnapshot = JobSnapshot(
            job_id=self._job_id,
            status=JobStatus.QUEUED,
            input_blocks=len(self._blocks),
            created_at=time.monotonic(),
        )
        # Cooperative-cancellation flags. Both make the pipeline's
        # ``is_cancelled`` predicate return True so the worker exits promptly;
        # which one was set determines the terminal status assigned by
        # :meth:`_execute`.
        self._cancel_requested: bool = False
        self._timed_out: bool = False
        self._started_once: bool = False

    # ------------------------------------------------------------------ #
    # read-only views
    # ------------------------------------------------------------------ #
    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def input_blocks(self) -> int:
        return len(self._blocks)

    @property
    def status(self) -> JobStatus:
        """Current :class:`JobStatus` (a consistent point-in-time read)."""
        with self._lock:
            return self._snapshot.status

    @property
    def result(self) -> DistillResult | None:
        """The :class:`~sparksage.distill.DistillResult`, or ``None`` if not done."""
        with self._lock:
            return self._snapshot.result

    def snapshot(self) -> JobSnapshot:
        """Return an immutable point-in-time :class:`JobSnapshot` copy.

        Safe to hand to another thread or a polling client without copying --
        the snapshot is frozen and never mutated in place.
        """
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------ #
    # cancellation
    # ------------------------------------------------------------------ #
    def cancel(self) -> bool:
        """Request cooperative cancellation of a running job.

        Returns whether the request was *accepted*. A job that has already
        reached a terminal state cannot be cancelled (returns ``False``); a
        queued or running job flips its cancel flag, which the pipeline's
        ``is_cancelled`` predicate polls at iteration boundaries -- the worker
        thread exits at the next boundary and the job resolves to
        :attr:`JobStatus.CANCELLED` with whatever partial result was computed.

        This is *cooperative*: Python cannot kill the worker thread. The
        pipeline finishes the current iteration (and any in-flight LLM merge
        call) before observing the flag. The partial result is retained on the
        snapshot for inspection.
        """
        with self._lock:
            if self._snapshot.status.is_terminal:
                return False
            self._cancel_requested = True
        return True

    def _is_cancelled(self) -> bool:
        """Predicate polled by the pipeline (true on cancel OR timeout)."""
        return self._cancel_requested or self._timed_out

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def run_sync(self) -> DistillResult:
        """Run the pipeline to completion in the *current* thread and return the result.

        Use this in scripts, REPLs, or any non-asyncio context. It blocks the
        caller for the whole run; subscribe via ``on_progress=`` for visibility.
        Raises :class:`RuntimeError` if the job has already been started or if
        it was cancelled/timed out before producing a usable result.
        """
        self._mark_running()
        self._execute()
        with self._lock:
            snap = self._snapshot
        if snap.result is None:
            raise RuntimeError(
                f"distill job {self._job_id} finished in status {snap.status.value} "
                "with no result"
            )
        return snap.result

    async def start(
        self,
        *,
        timeout: float | None = DEFAULT_JOB_TIMEOUT,
    ) -> DistillResult:
        """Run the pipeline in a worker thread and asynchronously await completion.

        The pipeline's blocking I/O (LLM merges, embedding) runs in a thread via
        :func:`asyncio.to_thread`, leaving the event loop free. The job's
        ``on_progress`` callback (driven by the pipeline) updates the snapshot
        from the worker thread.

        Parameters
        ----------
        timeout:
            Optional deadline in seconds. On expiry the job resolves to
            :attr:`JobStatus.TIMEOUT`, the cooperative-cancel predicate is
            flipped so the worker thread winds down at the next iteration
            boundary, and :class:`asyncio.TimeoutError` is raised to the
            caller. Any partial result is retained on the snapshot. ``None``
            (default) waits forever.

        Raises
        ------
        asyncio.TimeoutError
            If ``timeout`` elapses before the pipeline finishes.
        RuntimeError
            If the job was already started, or if it finished without producing
            a usable result.
        """
        self._mark_running()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._execute),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            with self._lock:
                self._timed_out = True
            # The worker thread keeps running; the predicate above will make it
            # exit at the next iteration boundary. Wait briefly for that wind-
            # down so the snapshot reflects the partial result when we return.
            self._await_wind_down()
            raise
        with self._lock:
            snap = self._snapshot
        if snap.result is None:
            raise RuntimeError(
                f"distill job {self._job_id} finished in status {snap.status.value} "
                "with no result"
            )
        return snap.result

    async def wait(self, *, timeout: float | None = None) -> JobSnapshot:
        """Asynchronously poll until the job reaches a terminal state.

        Unlike :meth:`start`, this does *not* run the pipeline -- it assumes a
        worker is already running (e.g. started via a manager) and just waits
        for the snapshot to settle. Returns the terminal :class:`JobSnapshot`.
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            with self._lock:
                terminal = self._snapshot.status.is_terminal
            if terminal:
                return self.snapshot()
            if deadline is not None and time.monotonic() >= deadline:
                raise asyncio.TimeoutError(
                    f"timed out waiting for distill job {self._job_id}"
                )
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _mark_running(self) -> None:
        """Transition ``queued -> running`` exactly once, erroring on reuse."""
        with self._lock:
            if self._started_once:
                raise RuntimeError(
                    f"distill job {self._job_id} has already been started; a job "
                    "is single-use -- submit a new one for a fresh run"
                )
            self._started_once = True
            now = time.monotonic()
            self._snapshot = replace(
                self._snapshot,
                status=JobStatus.RUNNING,
                started_at=now,
                progress=replace(
                    self._snapshot.progress,
                    phase=JobStatus.RUNNING.value,
                ),
            )
            current = self._snapshot
        self._emit(current)

    def _on_pipeline_progress(self, progress: DistillProgress) -> None:
        """Pipeline -> job callback: update progress under the lock, then emit.

        Called from the worker thread. Cheap: one dataclass build, one locked
        replace, one callback dispatch. The external callback is invoked
        *outside* the lock so a slow callback cannot block the worker.
        """
        job_progress = _progress_from_pipeline(progress)
        with self._lock:
            if self._snapshot.status is not JobStatus.RUNNING:
                # Status already moved to terminal (timeout/cancel); ignore
                # late progress emissions from the winding-down worker.
                return
            self._snapshot = replace(
                self._snapshot,
                progress=job_progress,
            )
            current = self._snapshot
        self._emit(current)

    def _execute(self) -> None:
        """Run the pipeline to completion and set the terminal snapshot.

        Runs in the worker thread (async path) or the caller thread (sync
        path). Pre-condition: :meth:`_mark_running` has already been called
        (which enforces the single-use contract and flipped the status to
        ``running``); this method only does the work and the terminal write.
        """
        try:
            result = self._pipeline.run(
                self._blocks,
                on_progress=self._on_pipeline_progress,
                is_cancelled=self._is_cancelled,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure on the snapshot
            self._set_terminal(
                status=JobStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                result=None,
            )
            _logger.exception("distill job %s failed", self._job_id)
            return

        with self._lock:
            cancelled = self._cancel_requested
            timed_out = self._timed_out
        if timed_out:
            self._set_terminal(JobStatus.TIMEOUT, result=result)
        elif cancelled:
            self._set_terminal(JobStatus.CANCELLED, result=result)
        else:
            self._set_terminal(JobStatus.SUCCESS, result=result)

    def _set_terminal(
        self,
        status: JobStatus,
        *,
        result: DistillResult | None,
        error: str | None = None,
    ) -> None:
        """Replace the snapshot with a terminal one and emit it (once)."""
        with self._lock:
            if self._snapshot.status.is_terminal:
                # Already terminal (e.g. timeout fired, then the worker wound
                # down and called _execute's success path). First writer wins;
                # the partial result is already on the snapshot.
                return
            now = time.monotonic()
            self._snapshot = replace(
                self._snapshot,
                status=status,
                finished_at=now,
                error=error,
                result=result,
                progress=replace(
                    self._snapshot.progress,
                    phase=JobStatus.RUNNING.value
                    if status is JobStatus.RUNNING
                    else "done",
                    percent=1.0,
                ),
            )
            current = self._snapshot
        self._emit(current)
        _logger.info(
            "distill job %s terminal: status=%s survivors=%d reduction=%.3f",
            self._job_id,
            status.value,
            len(result.survivors) if result is not None else 0,
            result.reduction if result is not None else 0.0,
        )

    def _await_wind_down(self, *, poll_interval: float = 0.05) -> None:
        """Block briefly until the worker thread finishes after a timeout/cancel.

        Bounded so a misbehaving pipeline that ignores ``is_cancelled`` cannot
        hang the caller forever -- after ``poll_interval * _WIND_DOWN_TICKS``
        ticks we return and let the snapshot reflect whatever is there.
        """
        for _ in range(_WIND_DOWN_TICKS):
            with self._lock:
                terminal = self._snapshot.status.is_terminal
            if terminal:
                return
            time.sleep(poll_interval)

    def _emit(self, snapshot: JobSnapshot) -> None:
        """Forward a snapshot to the external callback (outside the lock)."""
        callback = self._external_callback
        if callback is None:
            return
        try:
            callback(snapshot)
        except Exception:  # noqa: BLE001 - a buggy callback must not kill the worker
            _logger.exception(
                "distill job %s on_progress callback raised; ignoring",
                self._job_id,
            )

    def __repr__(self) -> str:
        with self._lock:
            status = self._snapshot.status
            job_id = self._job_id
            n = self._snapshot.input_blocks
        return f"DistillJob(job_id={job_id!r}, status={status.value!r}, input_blocks={n})"


#: Ticks to wait for the worker thread to wind down after a timeout/cancel
#: before returning. ``~10`` * ``0.05s`` = ``~0.5s`` headroom -- enough for a
#: well-behaved pipeline to observe ``is_cancelled`` at its next iteration
#: boundary without bogging down callers when the worker is genuinely blocked.
_WIND_DOWN_TICKS: int = 10


class JobManager:
    """In-process registry of :class:`DistillJob` s, keyed by id.

    The service-layer object a future ``/api/v1/distill`` route will wrap:

    * ``submit()`` returns a job (and its id) immediately -- the long-running
      work happens in a background thread;
    * ``snapshot(id)`` / ``get(id)`` back ``GET /api/v1/jobs/{id}``;
    * :meth:`wait_for` backs a long-poll or websocket flush.

    Jobs live in a plain dict. This is intentionally in-process: a single
    Distill run is CPU- and LLM-bound, not a distributed workload, and keeping
    the registry in-memory keeps the layer fully unit-testable with the
    deterministic fakes and zero infrastructure. When multi-process durability
    is needed (true crash recovery, horizontal scale), wrap this in a
    Redis-backed adapter -- the :class:`DistillJob` API stays unchanged.

    Parameters
    ----------
    pipeline:
        The :class:`DistillPipeline` used to run every submitted job. The
        pipeline's configuration (thresholds, clustering backend, candidate
        reducer) is shared across jobs; per-job overrides require constructing
        separate managers.

    Examples
    --------
    >>> import asyncio                                         # doctest: +SKIP
    >>> from sparksage import DistillPipeline, JobManager      # doctest: +SKIP
    >>> manager = JobManager(pipeline)                         # doctest: +SKIP
    >>> job = manager.submit(blocks)                           # doctest: +SKIP
    >>> result = asyncio.run(manager.wait_for(job.job_id))     # doctest: +SKIP
    """

    def __init__(self, pipeline: DistillPipeline) -> None:
        self._pipeline = pipeline
        self._jobs: dict[str, DistillJob] = {}
        self._lock = threading.Lock()

    @property
    def pipeline(self) -> DistillPipeline:
        return self._pipeline

    def submit(
        self,
        blocks: list[IdeaBlock],
        *,
        job_id: str | None = None,
        on_progress: JobProgressCallback | None = None,
        autostart: bool = True,
    ) -> DistillJob:
        """Register a new :class:`DistillJob` and (by default) start it in the background.

        Parameters
        ----------
        blocks, job_id, on_progress:
            Forwarded to :class:`DistillJob`.
        autostart:
            If ``True`` (default) the job starts running immediately in a
            background worker thread via :meth:`DistillJob.start` -- the call
            returns the moment the job is registered, without waiting. Set to
            ``False`` to obtain a queued job and start it yourself later with
            :meth:`DistillJob.start` / :meth:`DistillJob.run_sync`.

        Raises
        ------
        ValueError
            If ``job_id`` collides with an existing job in this manager.
        """
        job = DistillJob(
            blocks,
            self._pipeline,
            job_id=job_id,
            on_progress=on_progress,
        )
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job_id {job.job_id!r} already exists in this manager")
            self._jobs[job.job_id] = job

        if autostart:
            # Fire-and-forget: the asyncio loop driving wait_for() owns the
            # actual awaiting. We detach the task and let it surface any error
            # onto the job's snapshot (FAILED status) rather than raising here.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(
                    self._run_job_safely(job),
                    name=f"distill-job-{job.job_id}",
                )
            else:
                # No running loop (e.g. called from a sync context). Run on a
                # fresh loop in a daemon thread so submit() stays non-blocking.
                self._run_job_in_thread(job)
        return job

    def get(self, job_id: str) -> DistillJob | None:
        """Return the registered :class:`DistillJob`, or ``None`` if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        """Return the latest :class:`JobSnapshot` for ``job_id``, or ``None`` if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
        return job.snapshot() if job is not None else None

    def list_ids(self) -> list[str]:
        """All registered job ids, sorted by insertion order (deterministic for tests)."""
        with self._lock:
            return list(self._jobs.keys())

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._jobs

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    async def wait_for(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> DistillResult:
        """Asynchronously wait for ``job_id`` to finish and return its result.

        The job must have been started (typically via :meth:`submit`, which
        autostarts). Raises :class:`KeyError` if the id is unknown,
        :class:`asyncio.TimeoutError` if ``timeout`` elapses, or
        :class:`RuntimeError` if the job finished in a state without a usable
        result (``failed`` -- inspect :attr:`JobSnapshot.error`).
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown distill job_id: {job_id!r}")
        snap = await job.wait(timeout=timeout)
        if snap.result is None:
            raise RuntimeError(
                f"distill job {job_id} finished in status {snap.status.value} "
                f"with no result: {snap.error or '(no error detail)'}"
            )
        return snap.result

    async def gather(
        self,
        job_ids: list[str],
        *,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> dict[str, DistillResult | BaseException]:
        """Wait for several jobs at once, returning ``{job_id: result}``.

        With ``return_exceptions=True`` every outcome (including
        :class:`asyncio.TimeoutError` / :class:`RuntimeError`) is returned as
        the dict value rather than raised, so a single failing job does not
        abort the batch. ``timeout`` is applied to the whole gather, not per
        job.
        """
        coros = [self.wait_for(jid, timeout=timeout) for jid in job_ids]
        outcomes = await asyncio.gather(*coros, return_exceptions=return_exceptions)
        return dict(zip(job_ids, outcomes, strict=True))

    def forget(self, job_id: str) -> bool:
        """Drop a job from the registry. Returns whether it was present.

        The worker thread (if still running) is *not* cancelled -- call
        :meth:`DistillJob.cancel` first if you want it to wind down. This is a
        memory-management hook for long-lived managers that accumulate many
        finished jobs; drop only terminal jobs to avoid surprise.
        """
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    # ------------------------------------------------------------------ #
    # internals: autostart helpers
    # ------------------------------------------------------------------ #
    async def _run_job_safely(self, job: DistillJob) -> None:
        """Drive a job to completion on the running loop, surfacing failures on the snapshot."""
        try:
            await job.start()
        except asyncio.TimeoutError:
            # Already reflected on the snapshot as TIMEOUT; nothing to raise.
            pass
        except Exception:  # noqa: BLE001 - never let a background task die quietly
            _logger.exception(
                "autostarted distill job %s raised unexpectedly", job.job_id
            )

    def _run_job_in_thread(self, job: DistillJob) -> None:
        """Run a job on a fresh asyncio loop in a daemon thread (no running loop in caller)."""
        def _runner() -> None:
            try:
                job.run_sync()
            except Exception:  # noqa: BLE001 - snapshot already FAILED
                _logger.exception(
                    "autostarted distill job %s raised in background thread",
                    job.job_id,
                )

        thread = threading.Thread(
            target=_runner,
            name=f"distill-job-{job.job_id}",
            daemon=True,
        )
        thread.start()


__all__ = [
    "DEFAULT_JOB_TIMEOUT",
    "DistillJob",
    "JobManager",
    "JobProgress",
    "JobProgressCallback",
    "JobSnapshot",
    "JobStatus",
]
