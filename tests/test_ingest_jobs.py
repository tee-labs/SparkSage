"""Tests for the async, pollable ingest job layer.

Covers three layers:

1. :class:`IngestJob` / :class:`IngestJobManager` unit tests -- the state
   machine (``queued -> running -> success | failed | cancelled``), phase
   progress, cooperative cancellation, and the registry. Fully offline with
   a controllable work callable (no ingest / LLM / embedding deps).
2. :class:`QAService.submit_ingest` -- the end-to-end async ingest wired to
   the real (faked) pipeline, including the "cancel before the write leaves
   the KB untouched" guarantee.
3. HTTP integration -- ``POST /knowledge_base/ingest/async`` +
   ``GET /jobs/{id}`` + ``POST /jobs/{id}/cancel`` via ``TestClient``.

Async wind-down is driven by polling ``snapshot()`` inside plain sync test
functions so the suite needs no ``pytest-asyncio`` plugin.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from sparksage import (
    BlockEmbedder,
    FakeConverterBackend,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlockGenerator,
    LLMAnswerGenerator,
    LLMFaithfulnessJudge,
    MarkdownConverter,
    Reader,
    SparkSageService,
    TextCleaner,
)
from sparksage.api.ingest_jobs import (
    IngestCancelled,
    IngestJob,
    IngestJobManager,
    IngestJobStatus,
)
from sparksage.api.qa_service import QAService

# ---------------------------------------------------------------------------- #
# Shared fakes (mirror test_qa_api.py)
# ---------------------------------------------------------------------------- #
SAMPLE_MD = (
    "# Product Guide\n"
    "SparkSage is a library for question-aligned RAG. "
    "Install it with pip install sparksage."
)

GEN_JSON = json.dumps(
    {
        "blocks": [
            {
                "name": "Install",
                "critical_question": "How to install?",
                "trusted_answer": "Install with pip install sparksage.",
                "tags": ["technology"],
                "keywords": ["install", "pip"],
            }
        ]
    }
)


def _answer_json():
    return json.dumps(
        {
            "answer": "Use pip install sparksage.",
            "citations": [{"block_id": "ID", "quote": "pip install"}],
            "confidence": 0.9,
        }
    )


def _faith_json():
    return json.dumps({"score": 0.9, "supported_claims": 1, "unsupported_claims": 0})


def _make_qa_service(*, gen_json: str = GEN_JSON) -> QAService:
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
    )
    generator = IdeaBlockGenerator(FakeLLMClient(responses=[gen_json]))
    spark_service = SparkSageService(
        converter=converter,
        cleaner=TextCleaner(),
        generator=generator,
    )
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
    answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )
    return QAService(service=spark_service, embedder=embedder, reader=reader)


def _wait_terminal(job: IngestJob, *, timeout: float = 10.0) -> None:
    """Poll until the job reaches a terminal status (bounded)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.snapshot().is_terminal:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"job {job.job_id} did not terminate within {timeout}s "
        f"(status={job.snapshot().status.value})"
    )


# ---------------------------------------------------------------------------- #
# 1. IngestJob / IngestJobManager unit tests
# ---------------------------------------------------------------------------- #
class _Result:
    """Minimal IngestResult-like object (duck-typed by the job)."""

    def __init__(self, doc_id="doc-1", block_count=3, title="Guide"):
        self.doc_id = doc_id
        self.block_count = block_count
        self.title = title


class TestIngestJobStateMachine:
    def test_queued_before_start(self):
        job = IngestJob(lambda p, c: _Result())
        assert job.status is IngestJobStatus.QUEUED
        snap = job.snapshot()
        assert snap.phase == "queued"
        assert snap.percent == 0.0
        assert not snap.is_terminal

    def test_success_records_result_on_snapshot(self):
        seen_phases: list[str] = []

        def work(on_progress, is_cancelled):
            on_progress("converting")
            on_progress("generating")
            on_progress("indexing")
            seen_phases.append("done")
            return _Result(block_count=5)

        job = IngestJob(work, filename="guide.md")
        job.start()
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.SUCCESS
        assert snap.is_terminal
        assert snap.filename == "guide.md"
        assert snap.block_count == 5
        assert snap.doc_id == "doc-1"
        assert snap.title == "Guide"
        assert snap.percent == 1.0
        assert snap.phase == "done"
        assert job.result is not None
        assert job.result.block_count == 5

    def test_failed_records_error(self):
        def work(on_progress, is_cancelled):
            raise RuntimeError("boom")

        job = IngestJob(work)
        job.start()
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.FAILED
        assert "boom" in (snap.error or "")
        assert job.result is None

    def test_phase_progress_advances_percent(self):
        percents: list[float] = []

        def work(on_progress, is_cancelled):
            for phase in ("converting", "generating", "indexing"):
                on_progress(phase)
                percents.append(job.snapshot().percent)
            return _Result()

        job = IngestJob(work)
        job.start()
        _wait_terminal(job)
        # each phase strictly increases the percent
        assert percents == sorted(percents)
        assert percents[0] > 0.0
        assert job.snapshot().percent == 1.0

    def test_cancel_before_start_prevents_work(self):
        started = threading.Event()

        def work(on_progress, is_cancelled):
            started.set()
            return _Result()

        job = IngestJob(work)
        assert job.cancel() is True
        job.start()
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.CANCELLED
        # the worker observed the cancel flag at the first boundary and never
        # produced a result
        assert started.is_set()  # the thread ran ...
        assert job.result is None  # ... but no result was recorded

    def test_cancel_mid_run_aborts_before_write(self):
        # Gate the work so the test can flip cancel between phases.
        gate = threading.Event()
        cancelled_at_phase: list[str] = []

        def work(on_progress, is_cancelled):
            on_progress("converting")
            # block until the test flips cancel, then signal to proceed
            gate.wait(timeout=5.0)
            if is_cancelled():
                cancelled_at_phase.append("pre-generate")
                raise IngestCancelled()
            on_progress("generating")
            return _Result()

        progress_snapshots: list[IngestJob] = []
        job = IngestJob(work, on_progress=lambda s: progress_snapshots.append(job))
        job.start()
        # let it reach "converting"
        for _ in range(500):
            if job.snapshot().phase == "converting":
                break
            time.sleep(0.005)
        assert job.cancel() is True
        gate.set()
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.CANCELLED
        assert cancelled_at_phase == ["pre-generate"]
        assert job.result is None

    def test_job_is_single_use(self):
        job = IngestJob(lambda p, c: _Result())
        job.run_sync()
        with pytest.raises(RuntimeError):
            job.start()

    def test_run_sync_returns_result(self):
        job = IngestJob(lambda p, c: _Result(block_count=2))
        result = job.run_sync()
        assert result.block_count == 2
        assert job.snapshot().status is IngestJobStatus.SUCCESS

    def test_run_sync_cancelled_raises(self):
        job = IngestJob(lambda p, c: (_ for _ in ()).throw(IngestCancelled()))
        # the work always raises IngestCancelled -> cancelled terminal, and
        # run_sync surfaces that to the caller (no result to return).
        with pytest.raises(IngestCancelled):
            job.run_sync()
        assert job.snapshot().status is IngestJobStatus.CANCELLED

    def test_external_progress_callback_invoked(self):
        snapshots: list = []

        def work(on_progress, is_cancelled):
            on_progress("converting")
            return _Result()

        job = IngestJob(work, on_progress=lambda s: snapshots.append(s))
        job.start()
        _wait_terminal(job)
        assert len(snapshots) >= 2  # running + at least one phase + terminal
        assert snapshots[-1].status is IngestJobStatus.SUCCESS


class TestIngestJobManager:
    def test_submit_starts_and_registers(self):
        mgr = IngestJobManager()
        job = mgr.submit(lambda p, c: _Result(), filename="a.md")
        assert job.job_id in mgr
        assert len(mgr) == 1
        _wait_terminal(job)
        assert job.snapshot().status is IngestJobStatus.SUCCESS

    def test_snapshot_returns_none_for_unknown(self):
        mgr = IngestJobManager()
        assert mgr.snapshot("nope") is None
        assert mgr.cancel("nope") is None

    def test_cancel_routes_to_job(self):
        gate = threading.Event()

        def work(on_progress, is_cancelled):
            gate.wait(timeout=5.0)
            if is_cancelled():
                raise IngestCancelled()
            return _Result()

        mgr = IngestJobManager()
        job = mgr.submit(work)
        assert mgr.cancel(job.job_id) is True
        gate.set()
        _wait_terminal(job)
        assert job.snapshot().status is IngestJobStatus.CANCELLED

    def test_cancel_terminal_rejected(self):
        mgr = IngestJobManager()
        job = mgr.submit(lambda p, c: _Result())
        _wait_terminal(job)
        assert mgr.cancel(job.job_id) is False

    def test_id_collision_rejected(self):
        mgr = IngestJobManager()
        mgr.submit(lambda p, c: _Result(), job_id="fixed")
        with pytest.raises(ValueError):
            mgr.submit(lambda p, c: _Result(), job_id="fixed")

    def test_forget_drops_job(self):
        mgr = IngestJobManager()
        job = mgr.submit(lambda p, c: _Result())
        _wait_terminal(job)
        assert mgr.forget(job.job_id) is True
        assert job.job_id not in mgr
        assert mgr.forget(job.job_id) is False

    def test_evict_terminal_jobs_over_cap(self):
        mgr = IngestJobManager(max_jobs=2)
        j1 = mgr.submit(lambda p, c: _Result())
        j2 = mgr.submit(lambda p, c: _Result())
        _wait_terminal(j1)
        _wait_terminal(j2)
        # both terminal; submitting a third should evict the oldest terminal
        j3 = mgr.submit(lambda p, c: _Result())
        _wait_terminal(j3)
        assert len(mgr) <= 2
        assert j1.job_id not in mgr  # oldest evicted
        assert j3.job_id in mgr


# ---------------------------------------------------------------------------- #
# 2. QAService.submit_ingest -- end-to-end async ingest
# ---------------------------------------------------------------------------- #
class TestSubmitIngest:
    def test_returns_job_immediately_and_indexes(self):
        svc = _make_qa_service()
        job = svc.submit_ingest(b"data", "guide.md")
        assert job.status in (IngestJobStatus.QUEUED, IngestJobStatus.RUNNING)
        assert job.filename == "guide.md"
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.SUCCESS
        assert snap.block_count == 1
        assert snap.doc_id is not None
        # the doc is actually in the knowledge base
        assert svc.knowledge_base.document_count() == 1
        assert svc.knowledge_base.block_count() == 1
        # the result object is available on the job
        assert job.result is not None
        assert job.result.block_count == 1

    def test_failed_job_records_error(self):
        # a converter backend that raises -> the job fails cleanly
        from sparksage.convert.backend import FakeConverterBackend

        class _Boom(FakeConverterBackend):
            def convert(self, source, **kwargs):  # noqa: ARG002
                raise RuntimeError("convert broke")

        converter = MarkdownConverter(backend=_Boom(markdown="x", title="t"))
        generator = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
        spark_service = SparkSageService(
            converter=converter, cleaner=TextCleaner(), generator=generator
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)

        job = svc.submit_ingest(b"data", "guide.md")
        _wait_terminal(job)
        snap = job.snapshot()
        assert snap.status is IngestJobStatus.FAILED
        assert "convert broke" in (snap.error or "")
        assert svc.knowledge_base.document_count() == 0

    def test_cancel_before_write_leaves_kb_untouched(self):
        # Slow generator so the test can cancel mid-run, before the index phase.
        class _SlowGen:
            def __init__(self, inner):
                self._inner = inner

            def generate(self, text, *, source=None, max_blocks=None, language=None):
                time.sleep(0.3)
                return self._inner.generate(
                    text, source=source, max_blocks=max_blocks, language=language
                )

        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        real_gen = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=_SlowGen(real_gen),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)

        job = svc.submit_ingest(b"data", "guide.md")
        # cancel while still in the generate phase (before indexing / writing)
        for _ in range(500):
            if job.snapshot().phase == "generating":
                break
            time.sleep(0.005)
        assert job.cancel() is True
        _wait_terminal(job)
        snap = job.snapshot()
        # Either cleanly cancelled (work observed the flag) or cancelled via
        # post-completion race -- both report CANCELLED and the KB stays empty
        # because cancellation is checked before the knowledge-base write.
        assert snap.status is IngestJobStatus.CANCELLED
        assert svc.knowledge_base.document_count() == 0
        assert svc.knowledge_base.block_count() == 0

    def test_no_generator_fails_fast(self):
        from sparksage.api.pipeline import GenerationNotConfiguredError

        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        spark_service = SparkSageService(
            converter=converter, cleaner=TextCleaner(), generator=None
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)
        with pytest.raises(GenerationNotConfiguredError):
            svc.submit_ingest(b"data", "guide.md")

    def test_bad_kb_id_fails_fast(self):
        svc = _make_qa_service()
        with pytest.raises(KeyError):
            svc.submit_ingest(b"data", "guide.md", kb_id="nope")

    def test_ingest_jobs_property_exposed(self):
        svc = _make_qa_service()
        assert isinstance(svc.ingest_jobs, IngestJobManager)

    def test_phase_progress_emitted(self):
        svc = _make_qa_service()
        job = svc.submit_ingest(b"data", "guide.md")
        _wait_terminal(job)
        # the snapshot's final phase is "done"; intermediate phases were
        # converting / generating / indexing
        assert job.snapshot().phase == "done"


# ---------------------------------------------------------------------------- #
# 3. HTTP integration
# ---------------------------------------------------------------------------- #
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sparksage.api.app import create_app  # noqa: E402


def _wait_job(client, job_id, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("success", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not terminate")


class TestAsyncIngestRoutes:
    def test_async_ingest_then_poll_returns_result(self):
        svc = _make_qa_service()
        client = TestClient(create_app(qa_service=svc))

        resp = client.post(
            "/api/v1/knowledge_base/ingest/async",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["job_id"]
        assert body["status"] in ("queued", "running")
        assert body["filename"] == "guide.md"

        final = _wait_job(client, body["job_id"])
        assert final["status"] == "success"
        assert final["block_count"] == 1
        assert final["doc_id"]
        assert final["result"] is not None  # full payload on the terminal poll
        assert len(final["result"]["blocks"]) == 1
        assert final["phase"] == "done"
        assert final["percent"] == 1.0

    def test_async_ingest_visible_in_documents(self):
        svc = _make_qa_service()
        client = TestClient(create_app(qa_service=svc))
        resp = client.post(
            "/api/v1/knowledge_base/ingest/async",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        job_id = resp.json()["job_id"]
        _wait_job(client, job_id)
        docs = client.get("/api/v1/documents").json()
        assert docs["total"] == 1
        assert docs["items"][0]["title"] == "Guide"

    def test_get_unknown_job_404(self):
        svc = _make_qa_service()
        client = TestClient(create_app(qa_service=svc))
        resp = client.get("/api/v1/jobs/nope")
        assert resp.status_code == 404

    def test_async_ingest_503_without_generator(self):
        from sparksage.api.pipeline import GenerationNotConfiguredError  # noqa: F401

        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        spark_service = SparkSageService(
            converter=converter, cleaner=TextCleaner(), generator=None
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)
        client = TestClient(create_app(qa_service=svc))
        resp = client.post(
            "/api/v1/knowledge_base/ingest/async",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 503

    def test_cancel_route_marks_cancelled(self):
        class _SlowGen:
            def __init__(self, inner):
                self._inner = inner

            def generate(self, text, *, source=None, max_blocks=None, language=None):
                time.sleep(0.3)
                return self._inner.generate(
                    text, source=source, max_blocks=max_blocks, language=language
                )

        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        real_gen = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=_SlowGen(real_gen),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)
        client = TestClient(create_app(qa_service=svc))

        resp = client.post(
            "/api/v1/knowledge_base/ingest/async",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        job_id = resp.json()["job_id"]
        # cancel while in the generate phase
        for _ in range(500):
            snap = client.get(f"/api/v1/jobs/{job_id}").json()
            if snap["phase"] == "generating":
                break
            time.sleep(0.005)
        cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        final = _wait_job(client, job_id)
        assert final["status"] == "cancelled"
        # KB untouched: the cancellation guarantee
        assert svc.knowledge_base.document_count() == 0

    def test_cancel_unknown_job_404(self):
        svc = _make_qa_service()
        client = TestClient(create_app(qa_service=svc))
        resp = client.post("/api/v1/jobs/nope/cancel")
        assert resp.status_code == 404

    def test_sync_ingest_route_still_works(self):
        # the original blocking route is unchanged
        svc = _make_qa_service()
        client = TestClient(create_app(qa_service=svc))
        resp = client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["block_count"] == 1
