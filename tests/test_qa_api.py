"""Tests for the end-to-end QA service + HTTP routes.

Two layers are tested (mirroring ``test_api.py``):

1. **QAService unit tests** -- framework-agnostic, fully offline with
   :class:`FakeConverterBackend` / :class:`FakeLLMClient` /
   :class:`FakeEmbeddingClient`.
2. **HTTP integration tests** -- the QA routes via ``TestClient``.

The QA pipeline is: ingest (convert -> chunk -> index) -> query (retrieve ->
answer) -> feedback (record -> aggregate).
"""

from __future__ import annotations

import json
import time

import pytest

from sparksage import (
    BlockEmbedder,
    FakeConverterBackend,
    FakeEmbeddingClient,
    FakeLLMClient,
    FeedbackRating,
    IdeaBlockGenerator,
    LLMAnswerGenerator,
    LLMFaithfulnessJudge,
    MarkdownConverter,
    Reader,
    SparkSageService,
    TextCleaner,
)
from sparksage.api.pipeline import GenerationNotConfiguredError
from sparksage.api.qa_service import IngestResult, QAService

# ---------------------------------------------------------------------------- #
# Shared fakes
# ---------------------------------------------------------------------------- #
SAMPLE_MD = (
    "# Product Guide\n"
    "SparkSage is a library for question-aligned RAG. "
    "Install it with pip install sparksage. "
    "Deploy the API with uvicorn."
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
            },
            {
                "name": "Deploy",
                "critical_question": "How to deploy?",
                "trusted_answer": "Deploy the API with uvicorn.",
                "tags": ["process"],
                "keywords": ["deploy", "uvicorn"],
            },
        ]
    }
)


def _answer_json(text="Use pip install sparksage.", cid="ID"):
    return json.dumps(
        {
            "answer": text,
            "citations": [{"block_id": cid, "quote": "pip install"}],
            "confidence": 0.9,
        }
    )


def _faith_json(score=0.9):
    return json.dumps({"score": score, "supported_claims": 1, "unsupported_claims": 0})


SAMPLE_MD_V2 = (
    "# Product Guide v2\n"
    "SparkSage 2.0 ships question-aligned RAG with agentic QA. "
    "Install the new version with pip install --upgrade sparksage."
)

GEN_JSON_V2 = json.dumps(
    {
        "blocks": [
            {
                "name": "Install v2",
                "critical_question": "How to upgrade?",
                "trusted_answer": "Install with pip install --upgrade sparksage.",
                "tags": ["technology"],
                "keywords": ["upgrade", "install"],
            },
        ]
    }
)


class _ScriptedConverterBackend:
    """Returns pre-scripted (markdown, title) pairs in call order.

    Lets the update tests vary the *content* between ingest and update even
    though both uploads hit the same converter (the converter only sees the
    throwaway temp-file path, never the original filename).
    """

    def __init__(self, outputs):
        self._outputs = [tuple(o) for o in outputs]
        self.calls = 0

    def convert(self, source, **kwargs):
        md, title = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return md, title


def _make_qa_service(
    *,
    markdown: str = SAMPLE_MD,
    gen_json: str = GEN_JSON,
    answer_text: str = "Use pip install sparksage.",
    with_generator: bool = True,
    faith_score: float = 0.9,
    agent_controller=None,
    agent_max_iterations: int = 4,
) -> QAService:
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=markdown, title="Guide")
    )
    generator = (
        IdeaBlockGenerator(FakeLLMClient(responses=[gen_json]))
        if with_generator
        else None
    )
    spark_service = SparkSageService(
        converter=converter,
        cleaner=TextCleaner(),
        generator=generator,
    )
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))

    answer_client = FakeLLMClient(
        responses=[_answer_json(text=answer_text), _faith_json(faith_score)]
    )
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )
    return QAService(
        service=spark_service,
        embedder=embedder,
        reader=reader,
        agent_controller=agent_controller,
        agent_max_iterations=agent_max_iterations,
    )


# ---------------------------------------------------------------------------- #
# QAService.ingest_and_index
# ---------------------------------------------------------------------------- #
class TestIngestAndIndex:
    def test_returns_ingest_result(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md")
        assert isinstance(result, IngestResult)
        assert result.block_count == 2
        assert result.title == "Guide"
        assert result.doc_id

    def test_blocks_are_indexed_in_kb(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        assert svc.knowledge_base.block_count() == 2
        assert svc.knowledge_base.document_count() == 1

    def test_indexed_blocks_are_retrievable(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        res = svc.knowledge_base.search("install", k=2)
        assert len(res.chunks) > 0

    def test_tags_auto_extracted(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md", top_k=3)
        assert result.tags
        assert len(result.tags) <= 3

    def test_explicit_tags_kept(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(
            b"data", "guide.md", tags=["custom"], auto_tag=True
        )
        assert result.tags == ["custom"]

    def test_summary_produced(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md")
        assert result.summary is not None

    def test_no_generator_raises(self):
        svc = _make_qa_service(with_generator=False)
        with pytest.raises(GenerationNotConfiguredError):
            svc.ingest_and_index(b"data", "guide.md")

    def test_provenance_uses_filename(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "docs/report.pdf")
        assert result.source.uri == "docs/report.pdf"


class _SlowGen:
    """Wraps an IdeaBlockGenerator, sleeping inside generate()."""

    def __init__(self, inner, delay):
        self._inner = inner
        self._delay = delay

    def generate(self, text, *, source=None, max_blocks=None, language=None):
        time.sleep(self._delay)
        return self._inner.generate(
            text, source=source, max_blocks=max_blocks, language=language
        )


class _SlowSummarizer:
    def __init__(self, delay):
        self._delay = delay

    def summarize(self, text, *, max_sentences=3):
        time.sleep(self._delay)
        return "slow summary"


class TestIngestParallelism:
    def test_generate_and_summarize_overlap(self):
        # generate and summarize each sleep ~0.25s; if they ran serially the
        # ingest would take ~0.5s. Parallelism (option E) brings it under ~0.4s.
        delay = 0.25
        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        real_gen = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=_SlowGen(real_gen, delay),
            summarizer=_SlowSummarizer(delay),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)

        t0 = time.perf_counter()
        result = svc.ingest_and_index(b"data", "guide.md")
        elapsed = time.perf_counter() - t0

        assert result.block_count == 2
        assert elapsed < delay * 2 - 0.1, (
            f"ingest not parallelized: elapsed={elapsed:.2f}s >= ~{(delay * 2):.2f}s"
        )


def _make_scripted_qa_service(
    outputs, gen_responses, *, with_generator: bool = True
) -> QAService:
    converter = MarkdownConverter(backend=_ScriptedConverterBackend(outputs))
    generator = (
        IdeaBlockGenerator(FakeLLMClient(responses=gen_responses))
        if with_generator
        else None
    )
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


class TestUpdateDocumentAndReindex:
    def test_content_change_reindexes_keeping_doc_id(self):
        svc = _make_scripted_qa_service(
            [(SAMPLE_MD, "Guide"), (SAMPLE_MD_V2, "Guide v2")],
            [GEN_JSON, GEN_JSON_V2],
        )
        first = svc.ingest_and_index(b"data", "guide.md")
        old_ids = {str(b.id) for b in first.blocks}
        assert svc.knowledge_base.block_count() == 2

        result = svc.update_document_and_reindex(first.doc_id, b"data2", "guide2.md")

        assert result.doc_id == first.doc_id
        assert result.title == "Guide v2"
        assert result.block_count == 1
        assert svc.knowledge_base.block_count() == 1
        assert {str(b.id) for b in result.blocks}.isdisjoint(old_ids)
        assert svc.knowledge_base.search("upgrade", k=3).chunks
        stored = svc.service.get_document(first.doc_id)
        assert stored is not None
        assert stored.body_markdown == SAMPLE_MD_V2
        assert stored.title == "Guide v2"

    def test_unchanged_content_skips_generation_and_patches_metadata(self):
        gen = FakeLLMClient(responses=[GEN_JSON])
        converter = MarkdownConverter(backend=_ScriptedConverterBackend([(SAMPLE_MD, "Guide")]))
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=IdeaBlockGenerator(gen),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)
        first = svc.ingest_and_index(b"data", "guide.md")
        gen_calls_after_ingest = len(gen.calls)
        old_ids = {str(b.id) for b in first.blocks}

        result = svc.update_document_and_reindex(
            first.doc_id, b"data", "guide.md", title="Renamed", tags=["ops"]
        )

        assert result.doc_id == first.doc_id
        assert result.title == "Renamed"
        assert result.tags == ["ops"]
        assert len(gen.calls) == gen_calls_after_ingest
        assert {str(b.id) for b in result.blocks} == old_ids
        assert svc.knowledge_base.block_count() == len(old_ids)

    def test_nonexistent_document_raises(self):
        svc = _make_scripted_qa_service([(SAMPLE_MD, "Guide")], [GEN_JSON])
        with pytest.raises(KeyError):
            svc.update_document_and_reindex("nope", b"data", "guide.md")

    def test_content_change_without_generator_raises(self):
        svc = _make_scripted_qa_service(
            [(SAMPLE_MD, "Guide"), (SAMPLE_MD_V2, "Guide v2")],
            [GEN_JSON, GEN_JSON_V2],
        )
        first = svc.ingest_and_index(b"data", "guide.md")
        svc._service._generator = None
        with pytest.raises(GenerationNotConfiguredError):
            svc.update_document_and_reindex(first.doc_id, b"data2", "guide2.md")

    def test_cross_kb_update_raises(self):
        svc = _make_scripted_qa_service([(SAMPLE_MD, "Guide")], [GEN_JSON])
        new_info = svc.create_knowledge_base("second")
        result = svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        with pytest.raises(KeyError):
            svc.update_document_and_reindex(result.doc_id, b"data", "guide.md")


# ---------------------------------------------------------------------------- #
# Regression: KB ingest must be visible in the document-management store
# (https://github.com/tee-labs/SparkSage/issues — "文档管理里什么都看不到")
# ---------------------------------------------------------------------------- #
class TestSharedDocumentStore:
    def test_kb_ingest_visible_in_service_store(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        docs = svc.service.list_documents()
        assert len(docs) == 1
        assert docs[0].title == "Guide"

    def test_kb_ingest_count_in_service_store(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        assert svc.service.count_documents() == 1

    def test_multi_kb_document_count_stays_scoped(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        new_info = svc.create_knowledge_base("second")
        svc.ingest_and_index(b"data", "guide2.md", kb_id=new_info.kb_id)
        assert svc.knowledge_base.document_count() == 1
        assert svc._kbs[new_info.kb_id].document_count() == 1
        assert svc.service.count_documents() == 2

    def test_cross_kb_remove_document_is_safe(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        result = svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        deleted = svc.knowledge_base.remove_document(result.doc_id)
        assert deleted is False
        assert svc.service.get_document(result.doc_id) is not None
        assert svc._kbs[new_info.kb_id].document_count() == 1

    def test_cross_kb_update_document_raises(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        result = svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        with pytest.raises(KeyError):
            svc.knowledge_base.update_document(result.doc_id)

    def test_kb_remove_also_removes_from_service_store(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md")
        assert svc.knowledge_base.remove_document(result.doc_id) is True
        assert svc.service.get_document(result.doc_id) is None
        assert svc.service.count_documents() == 0


# ---------------------------------------------------------------------------- #
# QAService.ask
# ---------------------------------------------------------------------------- #
class TestAsk:
    def test_returns_qa_result(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False)
        assert result.query == "How to install?"
        assert not result.abstained
        assert "pip" in result.text

    def test_retrieval_scoped_to_kb(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to deploy?", use_lexical=False)
        assert result.retrieval is not None
        assert len(result.retrieval.chunks) > 0

    def test_abstention_on_no_kb_content(self):
        svc = _make_qa_service()
        # ask before any ingest -> empty retrieval -> abstention
        result = svc.ask("anything?", use_lexical=False)
        assert result.abstained

    def test_citation_bound_to_block(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        svc2 = _make_qa_service(answer_text="Use pip install sparksage.")
        svc2.ingest_and_index(b"data", "guide.md")
        result = svc2.ask("How to install?", use_lexical=False)
        assert not result.abstained


# ---------------------------------------------------------------------------- #
# QAService.ask(mode="agent") -- the agentic QA loop
# ---------------------------------------------------------------------------- #
class TestAgentMode:
    def _controller(self, *, retrieve_query="How to deploy?", extra_retrieve=False):
        from sparksage.agent import LLMAgentController

        responses = [
            json.dumps(
                {"thought": "need deploy", "action": "retrieve", "query": retrieve_query}
            ),
        ]
        if extra_retrieve:
            responses.append(
                json.dumps(
                    {"thought": "more", "action": "retrieve", "query": "uvicorn deploy"}
                )
            )
        responses.append(json.dumps({"thought": "enough", "action": "synthesize"}))
        return LLMAgentController(FakeLLMClient(responses=responses))

    def test_agent_mode_returns_agent_result(self):
        svc = _make_qa_service(agent_controller=self._controller())
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False, mode="agent")
        from sparksage.agent import AgentResult

        assert isinstance(result, AgentResult)
        assert result.iterations == 1  # one extra retrieval beyond the seed
        assert len(result.steps) == 2  # seed + 1 controller retrieval
        assert result.abstained is False

    def test_agent_mode_records_history(self):
        svc = _make_qa_service(agent_controller=self._controller())
        svc.ingest_and_index(b"data", "guide.md")
        svc.ask("How to install?", use_lexical=False, mode="agent")
        turns, total = svc.list_history()
        assert total == 2  # user + assistant
        assistant = [t for t in turns if t.role.value == "assistant"][0]
        assert assistant.result is not None  # serialized AskResponse payload

    def test_agent_trajectory_serialized_in_response_and_history(self):
        from sparksage.api.schemas import _to_ask_response

        svc = _make_qa_service(agent_controller=self._controller())
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False, mode="agent")
        payload = _to_ask_response(result).model_dump(mode="json")
        # the agent trajectory must not be discarded by the serializer
        assert payload["mode"] == "agent"
        assert payload["iterations"] == result.iterations
        assert payload["aborted"] == result.aborted
        assert len(payload["steps"]) == len(result.steps)
        assert payload["steps"][0]["query"]  # seed sub-query
        assert payload["steps"][0]["retrieved_count"] >= 0
        # the persisted history shares the same serializer, so it benefits too
        turns, _ = svc.list_history()
        assistant = [t for t in turns if t.role.value == "assistant"][0]
        assert assistant.result["mode"] == "agent"
        assert len(assistant.result["steps"]) == len(result.steps)

    def test_agent_mode_without_controller_raises(self):
        svc = _make_qa_service()  # no agent_controller wired
        svc.ingest_and_index(b"data", "guide.md")
        with pytest.raises(RuntimeError):
            svc.ask("How to install?", use_lexical=False, mode="agent")

    def test_agent_enabled_flag(self):
        svc = _make_qa_service()
        assert svc.agent_enabled is False
        svc2 = _make_qa_service(agent_controller=self._controller())
        assert svc2.agent_enabled is True

    def test_default_mode_unchanged(self):
        svc = _make_qa_service(agent_controller=self._controller())
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False)  # default mode
        from sparksage.qa import QAResult

        assert isinstance(result, QAResult)  # not an AgentResult

    def test_default_mode_trajectory_fields_absent(self):
        # single-shot mode must not surface an agent trajectory
        from sparksage.api.schemas import _to_ask_response

        svc = _make_qa_service(agent_controller=self._controller())
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False)  # default mode
        payload = _to_ask_response(result).model_dump(mode="json")
        assert payload["mode"] == "default"
        assert payload["iterations"] is None
        assert payload["aborted"] is None
        assert payload["steps"] == []


# ---------------------------------------------------------------------------- #
# QAService feedback
# ---------------------------------------------------------------------------- #
class TestFeedback:
    def test_add_feedback_string_rating(self):
        svc = _make_qa_service()
        record = svc.add_feedback("q", "a", "positive")
        assert record.rating == FeedbackRating.POSITIVE
        assert record.kb_id == svc.knowledge_base.kb_id

    def test_add_feedback_enum_rating(self):
        svc = _make_qa_service()
        record = svc.add_feedback("q", "a", FeedbackRating.NEGATIVE)
        assert record.rating == FeedbackRating.NEGATIVE

    def test_feedback_stats(self):
        svc = _make_qa_service()
        svc.add_feedback("q1", "a1", "positive")
        svc.add_feedback("q2", "a2", "negative")
        stats = svc.feedback_stats()
        assert stats.total == 2
        assert stats.positive == 1
        assert stats.negative == 1
        assert stats.approval == 0.5


# ---------------------------------------------------------------------------- #
# QAService conversation history (the query log the Q&A page restores)
# ---------------------------------------------------------------------------- #
class TestHistory:
    def test_ask_records_user_and_assistant_turns(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        svc.ask("How to install?", use_lexical=False)
        page, total = svc.list_history()
        assert total == 2
        assert [t.role.value for t in page] == ["assistant", "user"]
        assert page[1].content == "How to install?"

    def test_assistant_turn_carries_serialized_answer(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        res = svc.ask("How to install?", use_lexical=False)
        page, _ = svc.list_history()
        assistant = page[0]
        assert assistant.result is not None
        assert assistant.result["query"] == res.query
        assert assistant.result["answer"] == res.text
        assert "citations" in assistant.result
        assert assistant.query == "How to install?"

    def test_history_scoped_per_kb(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        svc.ask("How to install?", use_lexical=False, kb_id=new_info.kb_id)
        # default KB has no history; the second KB has the turns
        page, total = svc.list_history()
        assert total == 0
        page, total = svc.list_history(kb_id=new_info.kb_id)
        assert total == 2

    def test_clear_history(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        svc.ask("How to install?", use_lexical=False)
        assert svc.clear_history() == 2
        _, total = svc.list_history()
        assert total == 0

    def test_clear_history_scoped_per_kb(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        svc.ask("How to install?", use_lexical=False, kb_id=new_info.kb_id)
        assert svc.clear_history(kb_id=new_info.kb_id) == 2
        # default KB untouched (still empty)
        _, total = svc.list_history()
        assert total == 0

    def test_abstention_still_recorded(self):
        svc = _make_qa_service()
        # ask before any ingest -> abstention, but the turn is still logged
        svc.ask("anything?", use_lexical=False)
        _, total = svc.list_history()
        assert total == 2


# ---------------------------------------------------------------------------- #
# QAService.knowledge_base_info
# ---------------------------------------------------------------------------- #
class TestKnowledgeBaseInfo:
    def test_info_before_ingest(self):
        svc = _make_qa_service()
        info = svc.knowledge_base_info()
        assert info["block_count"] == 0
        assert info["document_count"] == 0
        assert info["name"] == "default"

    def test_info_after_ingest(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        info = svc.knowledge_base_info()
        assert info["block_count"] == 2
        assert info["document_count"] == 1


# ---------------------------------------------------------------------------- #
# HTTP integration tests
# ---------------------------------------------------------------------------- #
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sparksage.api.app import create_app  # noqa: E402


@pytest.fixture
def qa_client():
    svc = _make_qa_service()
    app = create_app(qa_service=svc)
    return TestClient(app)


def _ingest(client, filename="guide.md"):
    """Helper: ingest a doc and return the response body."""
    resp = client.post(
        "/api/v1/knowledge_base/ingest",
        files={"file": (filename, b"data", "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _make_scripted_qa_client(outputs, gen_responses):
    """HTTP client + service over a scripted converter/generator pair."""
    svc = _make_scripted_qa_service(outputs, gen_responses)
    return TestClient(create_app(qa_service=svc)), svc


class TestKBIngestRoute:
    def test_ingest_returns_blocks(self, qa_client):
        body = _ingest(qa_client)
        assert body["block_count"] == 2
        assert len(body["blocks"]) == 2
        assert body["title"] == "Guide"
        assert body["source"]["uri"] == "guide.md"
        assert body["tags"]
        assert body["summary"]

    def test_ingest_with_explicit_tags(self, qa_client):
        resp = qa_client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
            data={"tags": "alpha,beta"},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["alpha", "beta"]

    def test_ingest_missing_file(self, qa_client):
        resp = qa_client.post("/api/v1/knowledge_base/ingest")
        assert resp.status_code == 422

    def test_ingest_503_without_generator(self):
        svc = _make_qa_service(with_generator=False)
        app = create_app(qa_service=svc)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 503

    def test_kb_ingest_visible_in_documents_route(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.get("/api/v1/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["title"] == "Guide"

    def test_kb_remove_also_removes_from_documents_route(self, qa_client):
        body = _ingest(qa_client)
        doc_id = body["doc_id"]
        qa_client.delete(f"/api/v1/knowledge_base/documents/{doc_id}")
        resp = qa_client.get("/api/v1/documents")
        assert resp.json()["total"] == 0


class TestKBUpdateRoute:
    def test_put_unchanged_content_patches_metadata(self, qa_client):
        body = _ingest(qa_client)
        doc_id = body["doc_id"]
        old_ids = {b["id"] for b in body["blocks"]}

        resp = qa_client.put(
            f"/api/v1/knowledge_base/documents/{doc_id}",
            files={"file": ("guide.md", b"data", "text/plain")},
            data={"title": "Renamed", "tags": "alpha,beta"},
        )

        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["doc_id"] == doc_id
        assert out["title"] == "Renamed"
        assert out["tags"] == ["alpha", "beta"]
        assert {b["id"] for b in out["blocks"]} == old_ids

    def test_put_content_change_reindexes(self):
        client, svc = _make_scripted_qa_client(
            [(SAMPLE_MD, "Guide"), (SAMPLE_MD_V2, "Guide v2")],
            [GEN_JSON, GEN_JSON_V2],
        )
        body = _ingest(client)
        doc_id = body["doc_id"]
        old_ids = {b["id"] for b in body["blocks"]}

        resp = client.put(
            f"/api/v1/knowledge_base/documents/{doc_id}",
            files={"file": ("guide2.md", b"data2", "text/plain")},
        )

        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["doc_id"] == doc_id
        assert out["title"] == "Guide v2"
        assert out["block_count"] == 1
        assert {b["id"] for b in out["blocks"]}.isdisjoint(old_ids)
        assert svc.knowledge_base.block_count() == 1
        doc_resp = client.get("/api/v1/documents")
        assert doc_resp.status_code == 200
        assert doc_resp.json()["total"] == 1

    def test_put_nonexistent_document_404(self, qa_client):
        resp = qa_client.put(
            "/api/v1/knowledge_base/documents/nope",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 404

    def test_put_content_change_503_without_generator(self):
        client, svc = _make_scripted_qa_client(
            [(SAMPLE_MD, "Guide"), (SAMPLE_MD_V2, "Guide v2")],
            [GEN_JSON, GEN_JSON_V2],
        )
        body = _ingest(client)
        svc._service._generator = None
        resp = client.put(
            f"/api/v1/knowledge_base/documents/{body['doc_id']}",
            files={"file": ("guide2.md", b"data2", "text/plain")},
        )
        assert resp.status_code == 503


class TestIngestRouteDoesNotBlockEventLoop:
    """The ingest route offloads blocking work via asyncio.to_thread.

    Before that fix an ``async def`` route ran the blocking
    ``ingest_and_index`` (multiple second-long LLM calls) *on* the event-loop
    thread, so a single in-flight ingest froze the whole server -- even a
    trivial ``GET /api/v1/health`` could not answer until it finished.
    """

    def test_health_answers_during_slow_ingest(self):
        import asyncio

        from httpx import ASGITransport, AsyncClient

        delay = 0.5
        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        real_gen = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=_SlowGen(real_gen, delay),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark_service, embedder=embedder, reader=reader)
        app = create_app(qa_service=svc)

        async def run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                ingest_task = asyncio.create_task(
                    client.post(
                        "/api/v1/knowledge_base/ingest",
                        files={"file": ("g.md", b"data", "text/plain")},
                    )
                )
                await asyncio.sleep(0.05)
                t0 = time.perf_counter()
                health_resp = await client.get("/api/v1/health")
                health_elapsed = time.perf_counter() - t0
                ingest_resp = await ingest_task
                return health_resp, ingest_resp, health_elapsed

        health_resp, ingest_resp, health_elapsed = asyncio.run(run())
        assert ingest_resp.status_code == 200
        assert health_resp.status_code == 200
        assert health_elapsed < delay, (
            f"event loop was blocked: /health took {health_elapsed:.2f}s during a "
            f"{delay}s ingest (asyncio.to_thread should have offloaded it)"
        )


class TestQueryRoute:
    def test_ask_returns_answer(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "How to install?"
        assert not body["abstained"]
        assert "pip" in body["answer"]
        assert body["retrieved"]

    def test_ask_abstains_on_empty_kb(self, qa_client):
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "anything?", "use_lexical": False},
        )
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True

    def test_ask_with_tag_filter(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={
                "query": "install",
                "use_lexical": False,
                "tags": ["technology"],
            },
        )
        assert resp.status_code == 200

    def test_ask_missing_query_body(self, qa_client):
        resp = qa_client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_ask_with_history(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={
                "query": "how?",
                "use_lexical": False,
                "history": [
                    {"role": "user", "content": "how to install?"},
                ],
            },
        )
        assert resp.status_code == 200


class TestAgentQueryRoute:
    def test_agent_mode_returns_answer(self):
        from sparksage.agent import LLMAgentController

        ctrl = LLMAgentController(
            FakeLLMClient(
                responses=[
                    json.dumps(
                        {
                            "thought": "need deploy",
                            "action": "retrieve",
                            "query": "How to deploy?",
                        }
                    ),
                    json.dumps({"thought": "enough", "action": "synthesize"}),
                ]
            )
        )
        svc = _make_qa_service(agent_controller=ctrl)
        client = TestClient(create_app(qa_service=svc))
        _ingest(client)
        resp = client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False, "mode": "agent"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["query"] == "How to install?"
        assert body["abstained"] is False
        assert "pip" in body["answer"]
        assert body["retrieved"]  # accumulated evidence surfaced

    def test_agent_mode_without_controller_is_502(self):
        svc = _make_qa_service()  # no agent_controller
        client = TestClient(create_app(qa_service=svc))
        _ingest(client)
        resp = client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False, "mode": "agent"},
        )
        assert resp.status_code == 502


class TestKnowledgeBaseRoute:
    def test_kb_info(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.get("/api/v1/knowledge_base")
        assert resp.status_code == 200
        body = resp.json()
        assert body["block_count"] == 2
        assert body["document_count"] == 1
        assert body["name"] == "default"

    def test_remove_document(self, qa_client):
        body = _ingest(qa_client)
        doc_id = body["doc_id"]
        resp = qa_client.delete(f"/api/v1/knowledge_base/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # block count should drop to 0
        resp = qa_client.get("/api/v1/knowledge_base")
        assert resp.json()["block_count"] == 0

    def test_remove_document_404(self, qa_client):
        resp = qa_client.delete("/api/v1/knowledge_base/documents/missing")
        assert resp.status_code == 404


class TestFeedbackRoute:
    def test_record_feedback(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={
                "query": "How to install?",
                "answer_text": "pip install sparksage",
                "rating": "positive",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback_id"]
        assert body["rating"] == "positive"
        assert body["acknowledged"] is True

    def test_feedback_stats(self, qa_client):
        qa_client.post(
            "/api/v1/feedback",
            json={"query": "q1", "answer_text": "a", "rating": "positive"},
        )
        qa_client.post(
            "/api/v1/feedback",
            json={"query": "q2", "answer_text": "a", "rating": "negative"},
        )
        resp = qa_client.get("/api/v1/feedback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["positive"] == 1
        assert body["negative"] == 1
        assert body["approval"] == 0.5

    def test_invalid_rating(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={"query": "q", "answer_text": "a", "rating": "bogus"},
        )
        assert resp.status_code == 422

    def test_feedback_with_correction(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={
                "query": "q",
                "answer_text": "wrong",
                "rating": "corrected",
                "correction": "right answer",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["rating"] == "corrected"


class TestQueryHistoryRoute:
    def test_history_after_ask(self, qa_client):
        _ingest(qa_client)
        qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        resp = qa_client.get("/api/v1/query/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert [i["role"] for i in body["items"]] == ["assistant", "user"]
        assert body["items"][1]["content"] == "How to install?"

    def test_history_item_includes_answer_payload(self, qa_client):
        _ingest(qa_client)
        qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        body = qa_client.get("/api/v1/query/history").json()
        assistant = body["items"][0]
        assert assistant["result"] is not None
        assert assistant["result"]["query"] == "How to install?"
        assert "answer" in assistant["result"]
        assert "citations" in assistant["result"]

    def test_history_empty_before_any_ask(self, qa_client):
        resp = qa_client.get("/api/v1/query/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_clear_history(self, qa_client):
        _ingest(qa_client)
        qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        resp = qa_client.delete("/api/v1/query/history")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 2
        assert qa_client.get("/api/v1/query/history").json()["total"] == 0


# ---------------------------------------------------------------------------- #
# create_app without qa_service should NOT mount QA routes
# ---------------------------------------------------------------------------- #
class TestRouteMounting:
    def test_qa_routes_absent_without_qa_service(self):
        """When no qa_service is passed, the QA routes are not mounted."""
        from sparksage.clean.cleaner import TextCleaner
        from sparksage.convert.backend import FakeConverterBackend
        from sparksage.convert.converter import MarkdownConverter

        svc = SparkSageService(
            converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
            cleaner=TextCleaner(),
        )
        app = create_app(service=svc)
        client = TestClient(app)

        resp = client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 404

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 404

    def test_env_var_auto_builds_qa_service(self, monkeypatch):
        """SPARKSAGE_ENABLE_QA=1 auto-builds a QA service and mounts QA routes."""
        import sparksage.api.app as app_module

        qa_svc = _make_qa_service()
        monkeypatch.setenv("SPARKSAGE_ENABLE_QA", "1")
        monkeypatch.setattr(app_module, "build_qa_service", lambda: qa_svc)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 200
        assert app.state.qa_service is qa_svc

    def test_env_var_falsy_does_not_mount_qa_routes(self, monkeypatch):
        """When SPARKSAGE_ENABLE_QA is unset, QA routes stay unmounted."""
        import sparksage.api.app as app_module
        from sparksage.clean.cleaner import TextCleaner
        from sparksage.convert.backend import FakeConverterBackend
        from sparksage.convert.converter import MarkdownConverter

        monkeypatch.delenv("SPARKSAGE_ENABLE_QA", raising=False)
        base_svc = SparkSageService(
            converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
            cleaner=TextCleaner(),
        )
        monkeypatch.setattr(
            app_module, "build_default_service", lambda: base_svc
        )
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------- #
# Multi-knowledge-base management
# ---------------------------------------------------------------------------- #
class TestMultiKnowledgeBaseService:
    """QAService-level tests for create / list / delete + kb_id routing."""

    def test_create_knowledge_base(self):
        svc = _make_qa_service()
        info = svc.create_knowledge_base("ops-docs", description="ops")
        assert info.name == "ops-docs"
        assert info.kb_id in svc.kb_store

    def test_list_knowledge_bases_includes_default(self):
        svc = _make_qa_service()
        page, total = svc.list_knowledge_bases()
        assert total == 1
        assert page[0]["name"] == "default"

    def test_list_reflects_live_counts(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        page, _ = svc.list_knowledge_bases()
        assert page[0]["block_count"] == 2
        assert page[0]["document_count"] == 1

    def test_ingest_routes_to_specified_kb(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        # default KB stays empty
        assert svc.knowledge_base.block_count() == 0
        # second KB has the blocks
        second = svc._kbs[new_info.kb_id]
        assert second.block_count() == 2

    def test_ask_routes_via_kb_id(self):
        svc = _make_qa_service()
        new_info = svc.create_knowledge_base("second")
        svc.ingest_and_index(b"data", "guide.md", kb_id=new_info.kb_id)
        # asking the (empty) default KB abstains
        res_default = svc.ask("How to install?", use_lexical=False)
        assert res_default.abstained
        # asking the second KB returns an answer
        res_second = svc.ask(
            "How to install?", use_lexical=False, kb_id=new_info.kb_id
        )
        assert not res_second.abstained

    def test_delete_knowledge_base(self):
        svc = _make_qa_service()
        info = svc.create_knowledge_base("to-drop")
        assert svc.delete_knowledge_base(info.kb_id) is True
        assert info.kb_id not in svc.kb_store

    def test_cannot_delete_last_kb(self):
        svc = _make_qa_service()
        with pytest.raises(ValueError):
            svc.delete_knowledge_base(svc.active_kb_id)

    def test_delete_falls_back_active(self):
        svc = _make_qa_service()
        first = svc.active_kb_id
        svc.create_knowledge_base("second", set_active=True)
        svc.delete_knowledge_base(svc.active_kb_id)
        # active falls back to the remaining KB
        assert svc.active_kb_id == first

    def test_set_active_knowledge_base(self):
        svc = _make_qa_service()
        info = svc.create_knowledge_base("primary")
        svc.set_active_knowledge_base(info.kb_id)
        assert svc.active_kb_id == info.kb_id
        assert svc.knowledge_base.kb_id == info.kb_id

    def test_blocks_scoped_per_kb(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        new_info = svc.create_knowledge_base("second")
        page, total = svc.list_blocks(kb_id=new_info.kb_id)
        assert total == 0
        page, total = svc.list_blocks()
        assert total == 2


class TestMultiKnowledgeBaseRoutes:
    """HTTP tests for /api/v1/knowledge_bases + kb_id propagation."""

    def test_list_knowledge_bases(self, qa_client):
        resp = qa_client.get("/api/v1/knowledge_bases")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["name"] == "default"
        assert body["items"][0]["active"] is True

    def test_create_knowledge_base(self, qa_client):
        resp = qa_client.post(
            "/api/v1/knowledge_bases",
            json={"name": "ops", "set_active": False},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "ops"
        assert body["active"] is False

    def test_get_knowledge_base_404(self, qa_client):
        resp = qa_client.get("/api/v1/knowledge_bases/missing")
        assert resp.status_code == 404

    def test_activate_knowledge_base(self, qa_client):
        created = qa_client.post(
            "/api/v1/knowledge_bases",
            json={"name": "second", "set_active": False},
        ).json()
        resp = qa_client.post(f"/api/v1/knowledge_bases/{created['kb_id']}/activate")
        assert resp.status_code == 200
        assert resp.json()["active"] is True

    def test_delete_knowledge_base(self, qa_client):
        created = qa_client.post(
            "/api/v1/knowledge_bases",
            json={"name": "second", "set_active": False},
        ).json()
        resp = qa_client.delete(f"/api/v1/knowledge_bases/{created['kb_id']}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_last_kb_returns_409(self, qa_client):
        listing = qa_client.get("/api/v1/knowledge_bases").json()
        default_id = listing["items"][0]["kb_id"]
        resp = qa_client.delete(f"/api/v1/knowledge_bases/{default_id}")
        assert resp.status_code == 409

    def test_ingest_into_specific_kb(self, qa_client):
        created = qa_client.post(
            "/api/v1/knowledge_bases",
            json={"name": "second", "set_active": False},
        ).json()
        kb_id = created["kb_id"]
        resp = qa_client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
            data={"kb_id": kb_id},
        )
        assert resp.status_code == 200
        # default KB info should still be empty
        info = qa_client.get("/api/v1/knowledge_base").json()
        assert info["block_count"] == 0
        # the second KB has the blocks
        blocks = qa_client.get(
            "/api/v1/knowledge_base/blocks", params={"kb_id": kb_id}
        ).json()
        assert blocks["total"] == 2

    def test_ask_with_kb_id(self, qa_client):
        created = qa_client.post(
            "/api/v1/knowledge_bases",
            json={"name": "second", "set_active": False},
        ).json()
        kb_id = created["kb_id"]
        qa_client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
            data={"kb_id": kb_id},
        )
        # querying the (empty) default KB abstains
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True
        # querying the second KB returns an answer
        resp = qa_client.post(
            "/api/v1/query",
            json={
                "query": "How to install?",
                "use_lexical": False,
                "kb_id": kb_id,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["abstained"] is False

    def test_ask_unknown_kb_404(self, qa_client):
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "x", "kb_id": "missing"},
        )
        assert resp.status_code == 404
