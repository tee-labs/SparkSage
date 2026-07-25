"""Tests for the async, pollable :class:`DistillJob` and :class:`JobManager`.

All tests run fully offline and dependency-free. They exercise the full state
machine (``queued -> running -> success | failed | timeout | cancelled``), the
progress-callback plumbing, cooperative cancellation, timeout, and the
:class:`JobManager` registry, using :class:`FakeEmbeddingClient` /
:class:`FakeLLMClient` plus a controllable blocking backend where timing
matters.

Async entry points (:meth:`DistillJob.start`, :meth:`JobManager.wait_for`) are
driven through :func:`asyncio.run` inside plain sync test functions so the
suite needs no ``pytest-asyncio`` plugin.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from sparksage import (
    BlockEmbedder,
    BlockMerger,
    Cluster,
    ClusteringBackend,
    DistillJob,
    DistillPipeline,
    FakeEmbeddingClient,
    FakeLLMClient,
    JobManager,
    JobStatus,
)
from sparksage.distill.job import JobSnapshot
from sparksage.schema.ideablock import IdeaBlock


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _merge_json() -> str:
    return json.dumps(
        {
            "name": "Canonical",
            "critical_question": "What is the canonical answer?",
            "trusted_answer": "A merged, concise, verified answer.",
            "tags": ["IMPORTANT"],
            "entities": [
                {"entity_name": "SparkSage", "entity_type": "PRODUCT", "aliases": ["ss"]}
            ],
            "keywords": ["merged", "dedup"],
            "reasoning": "merged duplicates",
        }
    )


def _make_block(name: str, answer: str) -> IdeaBlock:
    return IdeaBlock(
        name=name, critical_question=f"What is {name}?", trusted_answer=answer
    )


def _duplicate_corpus() -> list[IdeaBlock]:
    return [
        _make_block("Deploy1", "deploy sparksage locally fast now"),
        _make_block("Deploy2", "deploy sparksage locally fast quick"),
        _make_block("Cook", "chocolate cake recipe sugar eggs flour"),
    ]


def _pipeline(
    *,
    responses: list[str] | None = None,
    backend: ClusteringBackend | None = None,
    **kwargs: object,
) -> DistillPipeline:
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
    merger = BlockMerger(FakeLLMClient(responses=responses or [_merge_json()]))
    return DistillPipeline(  # type: ignore[arg-type]
        embedder=embedder,
        merger=merger,
        clustering_backend=backend,
        **kwargs,
    )


class _BlockingBackend:
    """:class:`ClusteringBackend` that blocks on an event the first ``n_block`` calls.

    Lets the tests deterministically pause the worker inside a known iteration
    so cancel/timeout fire mid-run. Returns singleton clusters (no merges) so
    the pipeline does useful-but-trivial work each round.
    """

    def __init__(self, event: threading.Event, *, block_calls: int = 1) -> None:
        self._event = event
        self._block_calls = block_calls
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def cluster(
        self,
        vectors: dict[str, list[float]],
        *,
        threshold: float = 0.5,
    ) -> list[Cluster]:
        with self._lock:
            self._calls += 1
            should_block = self._calls <= self._block_calls
        if should_block:
            self._event.wait(timeout=10.0)
        return [Cluster(members=(bid,), confidence=1.0) for bid in vectors]


class _RaisingBackend:
    """A backend that always raises -- used to drive the job into the ``failed`` state."""

    def cluster(
        self,
        vectors: dict[str, list[float]],
        *,
        threshold: float = 0.5,
    ) -> list[Cluster]:
        raise RuntimeError("boom from clustering backend")


# ---------------------------------------------------------------------------- #
# JobStatus
# ---------------------------------------------------------------------------- #
class TestJobStatus:
    def test_terminal_states(self):
        assert JobStatus.SUCCESS.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.TIMEOUT.is_terminal
        assert JobStatus.CANCELLED.is_terminal

    def test_non_terminal_states(self):
        assert not JobStatus.QUEUED.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert JobStatus.RUNNING.is_running
        assert not JobStatus.QUEUED.is_running


# ---------------------------------------------------------------------------- #
# DistillJob: synchronous execution
# ---------------------------------------------------------------------------- #
class TestRunSync:
    def test_success_path(self):
        pipe = _pipeline(start_threshold=0.4, max_iterations=1)
        job = DistillJob(_duplicate_corpus(), pipe)
        assert job.status is JobStatus.QUEUED
        result = job.run_sync()
        assert job.status is JobStatus.SUCCESS
        assert len(result.survivors) == 2
        snap = job.snapshot()
        assert snap.is_terminal
        assert snap.duration is not None and snap.duration >= 0.0
        assert snap.progress.percent == pytest.approx(1.0)

    def test_progress_callback_invoked(self):
        pipe = _pipeline(start_threshold=0.4, max_iterations=1)
        events: list[JobSnapshot] = []
        job = DistillJob(_duplicate_corpus(), pipe, on_progress=events.append)
        job.run_sync()
        statuses = [e.status for e in events]
        assert JobStatus.RUNNING in statuses
        assert events[-1].status is JobStatus.SUCCESS
        # progress should be non-decreasing in percent
        percents = [e.progress.percent for e in events]
        assert percents == sorted(percents)

    def test_failed_path_captures_error(self):
        pipe = _pipeline(backend=_RaisingBackend(), start_threshold=0.4, max_iterations=2)
        job = DistillJob(_duplicate_corpus(), pipe)
        # The pipeline raises from the backend -> the job resolves to FAILED
        # (run_sync raises RuntimeError because result is None).
        with pytest.raises(RuntimeError, match="failed"):
            job.run_sync()
        snap = job.snapshot()
        assert snap.status is JobStatus.FAILED
        assert snap.error is not None and "boom" in snap.error
        assert snap.result is None

    def test_single_use_rejects_second_start(self):
        pipe = _pipeline(start_threshold=0.4, max_iterations=1)
        job = DistillJob(_duplicate_corpus(), pipe)
        job.run_sync()
        with pytest.raises(RuntimeError, match="single-use"):
            job.run_sync()

    def test_cancel_before_start_resolves_cancelled(self):
        pipe = _pipeline(start_threshold=0.99, max_iterations=3)
        job = DistillJob(_duplicate_corpus(), pipe)
        assert job.cancel() is True  # request cancel while still queued
        result = job.run_sync()  # pipeline breaks at iter 1 boundary
        snap = job.snapshot()
        assert snap.status is JobStatus.CANCELLED
        # partial result is retained: every input block survives, nothing merged
        assert len(result.survivors) == 3
        assert result.merged_out == []
        assert result.reduction == pytest.approx(0.0)

    def test_cancel_on_terminal_job_returns_false(self):
        pipe = _pipeline(start_threshold=0.99, max_iterations=1)
        job = DistillJob(_duplicate_corpus(), pipe)
        job.run_sync()
        assert job.cancel() is False  # already terminal


# ---------------------------------------------------------------------------- #
# DistillJob: asynchronous execution
# ---------------------------------------------------------------------------- #
class TestRunAsync:
    def test_start_success(self):
        async def _driver() -> JobSnapshot:
            pipe = _pipeline(start_threshold=0.4, max_iterations=1)
            job = DistillJob(_duplicate_corpus(), pipe)
            result = await job.start()
            assert len(result.survivors) == 2
            return job.snapshot()

        snap = asyncio.run(_driver())
        assert snap.status is JobStatus.SUCCESS

    def test_start_with_timeout_raises(self):
        async def _driver() -> JobSnapshot:
            event = threading.Event()
            backend = _BlockingBackend(event)
            pipe = _pipeline(
                backend=backend, start_threshold=0.99, max_iterations=3
            )
            job = DistillJob(_duplicate_corpus(), pipe)
            try:
                await job.start(timeout=0.1)
            except asyncio.TimeoutError:
                pass
            else:  # pragma: no cover - defensive
                raise AssertionError("expected asyncio.TimeoutError")
            # The worker is still blocked in the backend; release it so the
            # cooperative-cancel predicate (flipped by the timeout) can wind
            # the pipeline down at the next iteration boundary.
            event.set()
            snap = await job.wait()
            return snap

        snap = asyncio.run(_driver())
        assert snap.status is JobStatus.TIMEOUT
        assert snap.result is not None  # partial result retained

    def test_start_cancel_mid_run(self):
        async def _driver() -> tuple[JobSnapshot, int]:
            event = threading.Event()
            backend = _BlockingBackend(event)
            pipe = _pipeline(
                backend=backend, start_threshold=0.99, max_iterations=3
            )
            job = DistillJob(_duplicate_corpus(), pipe)
            task = asyncio.create_task(job.start(timeout=10.0))
            # Wait for the worker to enter the blocking backend.
            while backend.calls == 0:
                await asyncio.sleep(0.005)
            assert job.cancel() is True  # request cancel while blocked
            event.set()  # release iter 1; iter 2 boundary observes cancel
            snap = await job.wait()
            await task  # surface any unexpected error from start()
            return snap, backend.calls

        snap, calls = asyncio.run(_driver())
        assert snap.status is JobStatus.CANCELLED
        # Only the first iteration ever ran the backend; iter 2 broke first.
        assert calls == 1

    def test_wait_times_out(self):
        async def _driver() -> None:
            pipe = _pipeline(start_threshold=0.99, max_iterations=1)
            job = DistillJob(_duplicate_corpus(), pipe)
            # wait() without ever starting the job -> never reaches terminal.
            with pytest.raises(asyncio.TimeoutError):
                await job.wait(timeout=0.05)

        asyncio.run(_driver())


# ---------------------------------------------------------------------------- #
# JobSnapshot
# ---------------------------------------------------------------------------- #
class TestJobSnapshot:
    def test_snapshot_is_frozen(self):
        pipe = _pipeline(start_threshold=0.99, max_iterations=1)
        job = DistillJob(_duplicate_corpus(), pipe)
        snap = job.snapshot()
        with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
            snap.status = JobStatus.SUCCESS  # type: ignore[misc]

    def test_queued_snapshot_shape(self):
        pipe = _pipeline()
        job = DistillJob(_duplicate_corpus(), pipe, job_id="job-1")
        snap = job.snapshot()
        assert snap.job_id == "job-1"
        assert snap.status is JobStatus.QUEUED
        assert snap.input_blocks == 3
        assert snap.started_at is None
        assert snap.finished_at is None
        assert snap.duration is None
        assert snap.progress.phase == JobStatus.QUEUED.value
        assert snap.progress.percent == pytest.approx(0.0)

    def test_progress_snapshot_carries_iteration_details(self):
        pipe = _pipeline(start_threshold=0.4, max_iterations=1)
        events: list[JobSnapshot] = []
        job = DistillJob(_duplicate_corpus(), pipe, on_progress=events.append)
        job.run_sync()
        # Find an iteration-end event (candidate_pairs populated).
        end_events = [
            e for e in events
            if e.status is JobStatus.RUNNING and e.progress.candidate_pairs >= 0
        ]
        assert end_events, "expected at least one iteration-end progress event"
        last = end_events[-1]
        assert last.progress.iteration >= 1
        assert last.progress.active_blocks >= 1


# ---------------------------------------------------------------------------- #
# JobManager
# ---------------------------------------------------------------------------- #
class TestJobManager:
    def test_submit_autostart_and_wait_for(self):
        async def _driver() -> JobSnapshot:
            pipe = _pipeline(start_threshold=0.4, max_iterations=1)
            manager = JobManager(pipe)
            job = manager.submit(_duplicate_corpus(), job_id="run-1")
            assert job.job_id == "run-1"
            assert "run-1" in manager
            assert len(manager) == 1
            assert manager.list_ids() == ["run-1"]
            result = await manager.wait_for("run-1", timeout=10.0)
            assert len(result.survivors) == 2
            return manager.snapshot("run-1")  # type: ignore[return-value]

        snap = asyncio.run(_driver())
        assert snap is not None
        assert snap.status is JobStatus.SUCCESS

    def test_submit_autostart_false_then_manual_start(self):
        async def _driver() -> JobSnapshot:
            pipe = _pipeline(start_threshold=0.4, max_iterations=1)
            manager = JobManager(pipe)
            job = manager.submit(_duplicate_corpus(), autostart=False)
            assert job.status is JobStatus.QUEUED
            result = await job.start()
            assert len(result.survivors) == 2
            return job.snapshot()

        snap = asyncio.run(_driver())
        assert snap.status is JobStatus.SUCCESS

    def test_submit_id_collision_raises(self):
        pipe = _pipeline()
        manager = JobManager(pipe)
        manager.submit([], job_id="dup", autostart=False)
        with pytest.raises(ValueError, match="already exists"):
            manager.submit([], job_id="dup", autostart=False)

    def test_get_unknown_returns_none(self):
        pipe = _pipeline()
        manager = JobManager(pipe)
        assert manager.get("nope") is None
        assert manager.snapshot("nope") is None

    def test_wait_for_unknown_raises_keyerror(self):
        async def _driver() -> None:
            pipe = _pipeline()
            manager = JobManager(pipe)
            with pytest.raises(KeyError):
                await manager.wait_for("missing")

        asyncio.run(_driver())

    def test_forget_removes_job(self):
        pipe = _pipeline()
        manager = JobManager(pipe)
        manager.submit([], job_id="x", autostart=False)
        assert "x" in manager
        assert manager.forget("x") is True
        assert "x" not in manager
        assert manager.forget("x") is False  # already gone

    def test_gather_collects_results(self):
        async def _driver() -> None:
            pipe = _pipeline(start_threshold=0.4, max_iterations=1)
            manager = JobManager(pipe)
            j1 = manager.submit(_duplicate_corpus(), job_id="a")
            j2 = manager.submit(_duplicate_corpus(), job_id="b")
            results = await manager.gather([j1.job_id, j2.job_id], timeout=10.0)
            assert set(results) == {"a", "b"}
            for r in results.values():
                assert len(r.survivors) == 2

        asyncio.run(_driver())

    def test_sync_submit_runs_in_background_thread(self):
        # submit() called with no running loop -> daemon thread runs the job.
        # Poll the snapshot synchronously until terminal. (The job may already
        # have finished by the time submit() returns -- the fakes are fast --
        # so we only assert the terminal outcome, not the intermediate state.)
        pipe = _pipeline(start_threshold=0.4, max_iterations=1)
        manager = JobManager(pipe)
        job = manager.submit(_duplicate_corpus(), job_id="bg")
        deadline = time.monotonic() + 10.0
        while not job.snapshot().is_terminal:
            assert time.monotonic() < deadline, "background job did not finish"
            time.sleep(0.01)
        snap = job.snapshot()
        assert snap.status is JobStatus.SUCCESS
        assert snap.result is not None
        assert len(snap.result.survivors) == 2
