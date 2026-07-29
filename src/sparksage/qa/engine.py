"""End-to-end QA engine: query understanding -> retrieval -> answer.

:class:`QAEngine` is the framework-agnostic orchestrator that finally makes
SparkSage an end-to-end *question-answering* core rather than a
preprocessing+dedup library. It wires together the three existing right-half
components built in this roadmap:

    user query
        -> QueryProcessor   (classify -> intercept -> rewrite)   [optional]
        -> intent routing   (intent -> kb_id scope)              [optional]
        -> Retriever        (hybrid recall -> fuse -> re-rank -> filter)
        -> retrieval grading (relevance -> refine -> re-retrieve) [optional]
        -> Reader           (generate -> judge faithfulness -> answer / abstain)
        -> QAResult

It owns no business logic itself -- each stage is a swappable protocol -- so it
is fully unit-testable offline with :class:`~sparksage.generator.FakeLLMClient`
and :class:`~sparksage.embed.FakeEmbeddingClient`. An optional
:class:`QACache` short-circuits the whole pipeline for near-duplicate repeat
queries (the Phase-3 semantic cache implements this protocol).

Three optional gates form the symmetric self-correction policy: the query-side
``min_confidence`` floor (:class:`~sparksage.query.QueryProcessor`), the
retrieval-side ``min_relevance`` floor (self-reflective loop), and the
answer-side ``min_faithfulness`` floor (:class:`~sparksage.reader.Reader`) --
each says "I don't know / try again" rather than degrade silently.

The engine is deliberately *not* wired to the web layer -- a future
``/api/v1/query`` route will be a thin FastAPI wrapper around
:meth:`QAEngine.ask`, exactly as :class:`~sparksage.api.SparkSageService` wraps
the ingest pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sparksage.query.classifier import IntentResult
from sparksage.query.context import ConversationContext
from sparksage.query.expander import IdentityExpander, QueryExpander
from sparksage.query.processor import QueryProcessor, QueryResult
from sparksage.query.refiner import IdentityRefiner, QueryRefiner
from sparksage.reader.orchestrator import AnswerResult, Reader
from sparksage.retrieve.grader import RelevanceResult, RetrievalGrader
from sparksage.retrieve.models import RetrievalFilter, RetrievalResult
from sparksage.retrieve.orchestrator import RetrievalConfig, Retriever
from sparksage.schema.enums import QueryIntent

_logger = logging.getLogger(__name__)

#: Default relevance floor below which the self-reflective loop refines + re-retrieves.
DEFAULT_MIN_RELEVANCE = 0.5

#: Default maximum number of refine + re-retrieve rounds (caps latency / cost).
DEFAULT_MAX_REFINE_ITERATIONS = 2


@runtime_checkable
class QACache(Protocol):
    """Optional semantic cache for the QA engine.

    A :meth:`lookup` that returns a cached :class:`QAResult` for a
    near-duplicate prior query short-circuits the whole pipeline (the biggest
    cost lever, since the LLM calls dominate). :meth:`store` records a fresh
    result keyed by the query. The Phase-3 :class:`~sparksage.query.SemanticCache`
    implements this protocol.
    """

    def lookup(self, query: str) -> QAResult | None:
        """Return a cached result for a near-duplicate query, else ``None``."""
        ...

    def store(self, query: str, result: QAResult) -> None:
        """Record ``result`` under ``query``."""
        ...


@dataclass
class QAResult:
    """The full outcome of one :meth:`QAEngine.ask` call.

    Attributes
    ----------
    query:
        The original user query.
    query_result:
        The :class:`~sparksage.query.processor.QueryResult` (intent + rewrite),
        or ``None`` when no :class:`QueryProcessor` is wired.
    retrieval:
        The :class:`~sparksage.retrieve.models.RetrievalResult`.
    answer:
        The :class:`~sparksage.reader.orchestrator.AnswerResult`.
    relevance:
        The :class:`~sparksage.retrieve.grader.RelevanceResult` from the
        self-reflective loop, or ``None`` when no grader is wired.
    refined_query:
        The last query the self-reflective loop refined to, or ``None`` when no
        refinement happened.
    iterations:
        Number of refine + re-retrieve rounds run (``0`` when single-pass).
    cached:
        ``True`` when this result was served from the :class:`QACache` without
        re-running retrieval / generation.
    """

    query: str
    query_result: QueryResult | None = None
    retrieval: RetrievalResult | None = None
    answer: AnswerResult | None = None
    relevance: RelevanceResult | None = None
    refined_query: str | None = None
    iterations: int = 0
    cached: bool = False

    @property
    def accepted(self) -> bool:
        """Whether the query passed interception (or no processor was wired)."""
        return self.query_result is None or self.query_result.accepted

    @property
    def abstained(self) -> bool:
        """Whether the reader abstained (or the query was rejected)."""
        if self.answer is not None:
            return self.answer.abstained
        return not self.accepted

    @property
    def text(self) -> str:
        """The surfaced answer text (generated or canned reply)."""
        if self.answer is not None:
            return self.answer.answer.text
        if self.query_result is not None and self.query_result.default_reply:
            return self.query_result.default_reply
        return ""

    @property
    def citations(self) -> list[Any]:
        """The grounded citations on the surfaced answer (empty if none)."""
        if self.answer is not None:
            return list(self.answer.answer.citations)
        return []


@dataclass
class IntentKBRouter:
    """Map a classified intent to a knowledge-base id (intent -> KB routing).

    The glue that finally connects the existing
    :class:`~sparksage.query.classifier.IntentClassifier` to the existing
    :class:`~sparksage.kb.KnowledgeBase` multi-tenant scoping. Wire an instance
    as the :class:`QAEngine` ``intent_router`` and the engine will set
    :attr:`~sparksage.retrieve.models.RetrievalFilter.kb_id` from the classified
    intent before retrieval -- so a financial query hits the finance KB and a
    support query hits the support KB automatically.

    Attributes
    ----------
    routing:
        Maps each :class:`~sparksage.schema.enums.QueryIntent` to a ``kb_id``.
    default:
        ``kb_id`` used when the intent is not in ``routing`` (``None`` = no
        scoping, fall back to the unfiltered corpus).
    fallback:
        Optional callable invoked when no explicit mapping and no ``default``
        apply. Lets callers encode richer routing (e.g. on confidence).

    It is callable, so it satisfies the ``intent_router`` contract directly::

        router = IntentKBRouter(routing={QueryIntent.FINANCIAL_DATA: "kb-fin"})
        engine = QAEngine(..., intent_router=router)
    """

    routing: dict[QueryIntent, str]
    default: str | None = None
    fallback: Callable[[IntentResult], str | None] | None = None

    def route(self, intent: IntentResult) -> str | None:
        kb_id = self.routing.get(intent.intent)
        if kb_id:
            return kb_id
        if self.default:
            return self.default
        if self.fallback is not None:
            try:
                return self.fallback(intent)
            except Exception as exc:  # pragma: no cover - defensive
                _logger.warning("intent_router fallback raised %s; no routing", exc)
                return None
        return None

    def __call__(self, intent: IntentResult) -> str | None:
        return self.route(intent)


class QAEngine:
    """End-to-end QA orchestrator: query -> retrieval -> answer.

    Parameters
    ----------
    retriever:
        The :class:`~sparksage.retrieve.Retriever` used for the retrieval stage.
    reader:
        The :class:`~sparksage.reader.Reader` used for the answer stage.
    query_processor:
        Optional :class:`~sparksage.query.QueryProcessor`. When ``None`` the raw
        query goes straight to retrieval (LLM-free query understanding).
    query_expander:
        Optional :class:`~sparksage.query.QueryExpander`. When wired (and the
        rewriter did not already emit sub-queries) the search query is expanded
        into ``n`` variants and RRF-fused -- the multi-query recall boost.
    cache:
        Optional :class:`QACache`. When wired, a cache hit returns immediately.
    intent_router:
        Optional ``Callable[[IntentResult], str | None]``. When wired (and a
        query processor classifies the intent), the returned ``kb_id`` is merged
        into the :class:`~sparksage.retrieve.RetrievalFilter` so retrieval is
        scoped to the right knowledge base -- connecting the existing
        :class:`~sparksage.query.IntentClassifier` to the existing
        :class:`~sparksage.kb.KnowledgeBase` multi-tenancy. A per-call
        ``filter.kb_id`` always wins. See :class:`IntentKBRouter`.
    retrieval_grader:
        Optional :class:`~sparksage.retrieve.RetrievalGrader`. When wired, each
        retrieval is graded for relevance; a low score triggers the
        self-reflective loop (refine query -> re-retrieve) -- the retrieval-side
        gate symmetric to the query-side ``min_confidence`` and answer-side
        ``min_faithfulness`` gates.
    query_refiner:
        Optional :class:`~sparksage.query.QueryRefiner`. When wired alongside a
        grader, a low relevance score produces a refined query and the engine
        re-retrieves (up to ``max_iterations`` rounds), keeping the best-graded
        result so refinement can never lower quality.
    min_relevance:
        Relevance score below which the self-reflective loop fires (default
        :data:`DEFAULT_MIN_RELEVANCE`).
    max_iterations:
        Cap on refine + re-retrieve rounds (default
        :data:`DEFAULT_MAX_REFINE_ITERATIONS`).
    config:
        :class:`~sparksage.retrieve.RetrievalConfig` defaults (``k`` /
        ``use_lexical`` / ``use_rerank``). Per-call kwargs override.

    Examples
    --------
    >>> from sparksage import FakeLLMClient, FakeEmbeddingClient   # doctest: +SKIP
    >>> from sparksage.qa import QAEngine                           # doctest: +SKIP
    >>> engine = QAEngine(retriever=..., reader=...)                # doctest: +SKIP
    >>> result = engine.ask("how to deploy")                       # doctest: +SKIP
    >>> result.text                                                 # doctest: +SKIP
    """

    def __init__(
        self,
        retriever: Retriever,
        reader: Reader,
        *,
        query_processor: QueryProcessor | None = None,
        query_expander: QueryExpander | None = None,
        cache: QACache | None = None,
        intent_router: Callable[[IntentResult], str | None] | None = None,
        retrieval_grader: RetrievalGrader | None = None,
        query_refiner: QueryRefiner | None = None,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        max_iterations: int = DEFAULT_MAX_REFINE_ITERATIONS,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._reader = reader
        self._query_processor = query_processor
        self._query_expander = query_expander
        self._cache = cache
        self._intent_router = intent_router
        if retrieval_grader is not None and not isinstance(
            retrieval_grader, RetrievalGrader
        ):
            raise TypeError(
                "retrieval_grader must implement the RetrievalGrader protocol"
            )
        if query_refiner is not None and not isinstance(query_refiner, QueryRefiner):
            raise TypeError("query_refiner must implement the QueryRefiner protocol")
        if intent_router is not None and not callable(intent_router):
            raise TypeError("intent_router must be callable")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be in [0.0, 1.0]")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an int")
        if max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        self._retrieval_grader = retrieval_grader
        self._query_refiner = query_refiner
        self._min_relevance = min_relevance
        self._max_iterations = max_iterations
        self._config = config if config is not None else RetrievalConfig()

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def reader(self) -> Reader:
        return self._reader

    @property
    def query_processor(self) -> QueryProcessor | None:
        return self._query_processor

    @property
    def query_expander(self) -> QueryExpander | None:
        return self._query_expander

    @property
    def cache(self) -> QACache | None:
        return self._cache

    @property
    def intent_router(self) -> Callable[[IntentResult], str | None] | None:
        return self._intent_router

    @property
    def retrieval_grader(self) -> RetrievalGrader | None:
        return self._retrieval_grader

    @property
    def query_refiner(self) -> QueryRefiner | None:
        return self._query_refiner

    @property
    def min_relevance(self) -> float:
        return self._min_relevance

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def ask(
        self,
        query: str,
        *,
        context: ConversationContext | None = None,
        filter: RetrievalFilter | None = None,
        k: int | None = None,
        use_lexical: bool | None = None,
        use_rerank: bool | None = None,
        use_cache: bool = True,
    ) -> QAResult:
        """Answer ``query`` end-to-end.

        Parameters
        ----------
        query:
            The raw user query.
        context:
            Optional :class:`~sparksage.query.ConversationContext` for
            multi-turn anaphora resolution (forwarded to the query processor).
        filter:
            Optional :class:`RetrievalFilter` metadata scope (forwarded to
            retrieval).
        k, use_lexical, use_rerank:
            Per-call retrieval overrides; default to the engine's
            :class:`RetrievalConfig`.
        use_cache:
            When ``True`` (default) and a cache is wired, a cache hit returns
            immediately and a miss stores the new result.
        """
        query = str(query)
        if use_cache and self._cache is not None:
            hit = self._cache.lookup(query)
            if hit is not None:
                return QAResult(query=query, cached=True, **_without_cached(hit))

        query_result: QueryResult | None = None
        if self._query_processor is not None:
            query_result = self._query_processor.process(query, context)
            if not query_result.accepted:
                result = QAResult(query=query, query_result=query_result)
                self._maybe_store(query, result, use_cache)
                return result

        scoped_filter = self._apply_intent_routing(filter, query_result)

        search_query = (
            query_result.rewrite.rewritten_query
            if query_result is not None
            else query
        )
        sub_queries = (
            query_result.rewrite.sub_queries
            if query_result is not None and query_result.rewrite.sub_queries
            else None
        )
        retrieval = self._retrieve(
            search_query,
            context=context,
            filter=scoped_filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            sub_queries=sub_queries,
        )

        relevance, retrieval, refined_query, iterations = self._reflective_retrieve(
            query,
            search_query,
            retrieval,
            context=context,
            filter=scoped_filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
        )

        answer = self._reader.answer(query, retrieval.chunks)
        result = QAResult(
            query=query,
            query_result=query_result,
            retrieval=retrieval,
            answer=answer,
            relevance=relevance,
            refined_query=refined_query,
            iterations=iterations,
        )
        self._maybe_store(query, result, use_cache)
        return result

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        search_query: str,
        *,
        context: ConversationContext | None,
        filter: RetrievalFilter | None,
        k: int | None,
        use_lexical: bool | None,
        use_rerank: bool | None,
        sub_queries: list[str] | None = None,
    ) -> RetrievalResult:
        """One retrieval pass, dispatching to the multi-query or single path.

        ``sub_queries`` (from the rewriter) take the RRF-fused multi-retrieve
        path; otherwise an :class:`~sparksage.query.QueryExpander` (if wired)
        supplies variants; otherwise a plain dense(+lexical) search.
        """
        if sub_queries:
            return self._multi_retrieve(
                search_query,
                sub_queries,
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
            )
        if self._query_expander is not None and not isinstance(
            self._query_expander, IdentityExpander
        ):
            variants = self._query_expander.expand(search_query)
            return self._multi_retrieve(
                search_query,
                variants[1:] if len(variants) > 1 else [],
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
            )
        return self._retriever.search(
            search_query,
            k=k or self._config.k,
            filter=filter,
            use_lexical=self._resolved(use_lexical, self._config.use_lexical),
            use_rerank=self._resolved(use_rerank, self._config.use_rerank),
        )

    def _apply_intent_routing(
        self,
        filter: RetrievalFilter | None,
        query_result: QueryResult | None,
    ) -> RetrievalFilter | None:
        """Merge an intent-derived ``kb_id`` into ``filter`` when a router is wired.

        A per-call ``filter.kb_id`` always wins (the caller knows best); the
        router only fills in a scope when none was set. Returns the original
        ``filter`` unchanged when routing does not apply.
        """
        if self._intent_router is None or query_result is None:
            return filter
        try:
            kb_id = self._intent_router(query_result.intent)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("intent_router raised %s; skipping routing", exc)
            return filter
        if not kb_id:
            return filter
        if filter is None:
            return RetrievalFilter(kb_id=kb_id)
        if filter.kb_id is None:
            from dataclasses import replace

            return replace(filter, kb_id=kb_id)
        return filter

    def _reflective_retrieve(
        self,
        query: str,
        search_query: str,
        retrieval: RetrievalResult,
        *,
        context: ConversationContext | None,
        filter: RetrievalFilter | None,
        k: int | None,
        use_lexical: bool | None,
        use_rerank: bool | None,
    ) -> tuple[RelevanceResult | None, RetrievalResult, str | None, int]:
        """Grade relevance; on a low score refine the query and re-retrieve.

        Returns ``(best_relevance, best_retrieval, last_refined_query, rounds)``.
        The best-graded retrieval is kept across rounds so refinement can never
        lower quality. No-op (returns the inputs untouched) when no grader is
        wired; grades but never re-retrieves when no real refiner is wired.
        """
        if self._retrieval_grader is None:
            return None, retrieval, None, 0

        best_relevance = self._retrieval_grader.grade(query, retrieval.chunks)
        best_retrieval = retrieval
        current_query = search_query
        refined_query: str | None = None
        rounds = 0

        can_refine = self._query_refiner is not None and not isinstance(
            self._query_refiner, IdentityRefiner
        )
        while (
            best_relevance.score < self._min_relevance
            and can_refine
            and rounds < self._max_iterations
        ):
            refined = self._query_refiner.refine(
                current_query, best_relevance.score, best_relevance.reasoning
            )
            refined = (refined or "").strip()
            rounds += 1
            if not refined or refined.lower() == current_query.strip().lower():
                break
            refined_query = refined
            new_retrieval = self._retrieve(
                refined,
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
            )
            new_relevance = self._retrieval_grader.grade(query, new_retrieval.chunks)
            if new_relevance.score > best_relevance.score:
                best_relevance = new_relevance
                best_retrieval = new_retrieval
                current_query = refined
            if best_relevance.score >= self._min_relevance:
                break

        return best_relevance, best_retrieval, refined_query, rounds

    def _multi_retrieve(
        self,
        primary: str,
        sub_queries: list[str],
        *,
        context: ConversationContext | None,
        filter: RetrievalFilter | None,
        k: int | None,
        use_lexical: bool | None,
        use_rerank: bool | None,
    ) -> RetrievalResult:
        """Run retrieval for the primary + each sub-query and RRF-fuse.

        This is the COMPARISON / multi-hop path: each sub-query gets its own
        recall, the ranked lists are fused, then the fused pool is re-ranked /
        filtered as usual. Sub-queries are exactly the
        :class:`~sparksage.query.rewriter.RewriteResult.sub_queries` the
        rewriter already emits -- finally consumed here.
        """
        from sparksage.retrieve.fusion import reciprocal_rank_fusion

        resolved_k = k or self._config.k
        lex = self._resolved(use_lexical, self._config.use_lexical)
        rr = self._resolved(use_rerank, self._config.use_rerank)

        queries = [primary] + [q for q in sub_queries if q and q != primary]
        if len(queries) == 1:
            return self._retriever.search(
                queries[0], k=resolved_k, filter=filter, use_lexical=lex, use_rerank=rr
            )

        fused_ids: dict[str, float] = {}
        registry = self._retriever._registry  # noqa: SLF001 (shared registry)
        dense_by: dict[str, float] = {}
        lex_by: dict[str, float] = {}
        rankings = []
        for q in queries:
            res = self._retriever.search(
                q, k=max(resolved_k * 2, 5), filter=filter, use_lexical=lex, use_rerank=False
            )
            rankings.append(res.dense_hits)
            for h in res.dense_hits:
                dense_by.setdefault(h.block_id, h.score)
            for h in res.lexical_hits:
                lex_by.setdefault(h.block_id, h.score)

        from sparksage.retrieve.models import RetrievedChunk

        if any(rankings):
            fused = reciprocal_rank_fusion(rankings, top_n=max(resolved_k * 2, 5))
        else:
            fused = []
        for h in fused:
            fused_ids[h.block_id] = h.score

        chunks = [
            RetrievedChunk(
                block=registry[bid],
                score=score,
                dense_score=dense_by.get(bid),
                lexical_score=lex_by.get(bid),
                rank=i,
            )
            for i, (bid, score) in enumerate(
                sorted(fused_ids.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            if bid in registry
        ][:resolved_k]

        return RetrievalResult(
            query=primary,
            chunks=chunks,
            dense_hits=[h for h in fused if h.block_id in dense_by] or fused,
            lexical_hits=[],
            fused=True,
            reranked=False,
            filtered_out=0,
        )

    @staticmethod
    def _resolved(call_value: bool | None, config_value: bool) -> bool:
        return config_value if call_value is None else call_value

    def _maybe_store(self, query: str, result: QAResult, use_cache: bool) -> None:
        if use_cache and self._cache is not None and result.answer is not None:
            try:
                self._cache.store(query, result)
            except Exception as exc:  # pragma: no cover - defensive
                _logger.warning("QACache.store failed: %s", exc)


def _without_cached(result: QAResult) -> dict[str, Any]:
    """Return QAResult fields except ``cached`` (so it can be re-set)."""
    return {
        "query_result": result.query_result,
        "retrieval": result.retrieval,
        "answer": result.answer,
        "relevance": result.relevance,
        "refined_query": result.refined_query,
        "iterations": result.iterations,
    }


__all__ = [
    "DEFAULT_MAX_REFINE_ITERATIONS",
    "DEFAULT_MIN_RELEVANCE",
    "IntentKBRouter",
    "QAEngine",
    "QACache",
    "QAResult",
]
