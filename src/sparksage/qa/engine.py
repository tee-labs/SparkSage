"""End-to-end QA engine: query understanding -> retrieval -> answer.

:class:`QAEngine` is the framework-agnostic orchestrator that finally makes
SparkSage an end-to-end *question-answering* core rather than a
preprocessing+dedup library. It wires together the three existing right-half
components built in this roadmap:

    user query
        -> QueryProcessor   (classify -> intercept -> rewrite)   [optional]
        -> Retriever        (hybrid recall -> fuse -> re-rank -> filter)
        -> Reader           (generate -> judge faithfulness -> answer / abstain)
        -> QAResult

It owns no business logic itself -- each stage is a swappable protocol -- so it
is fully unit-testable offline with :class:`~sparksage.generator.FakeLLMClient`
and :class:`~sparksage.embed.FakeEmbeddingClient`. An optional
:class:`QACache` short-circuits the whole pipeline for near-duplicate repeat
queries (the Phase-3 semantic cache implements this protocol).

The engine is deliberately *not* wired to the web layer -- a future
``/api/v1/query`` route will be a thin FastAPI wrapper around
:meth:`QAEngine.ask`, exactly as :class:`~sparksage.api.SparkSageService` wraps
the ingest pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sparksage.query.context import ConversationContext
from sparksage.query.expander import IdentityExpander, QueryExpander
from sparksage.query.processor import QueryProcessor, QueryResult
from sparksage.reader.orchestrator import AnswerResult, Reader
from sparksage.retrieve.models import RetrievalFilter, RetrievalResult
from sparksage.retrieve.orchestrator import RetrievalConfig, Retriever

_logger = logging.getLogger(__name__)


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
    cached:
        ``True`` when this result was served from the :class:`QACache` without
        re-running retrieval / generation.
    """

    query: str
    query_result: QueryResult | None = None
    retrieval: RetrievalResult | None = None
    answer: AnswerResult | None = None
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
        config: RetrievalConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._reader = reader
        self._query_processor = query_processor
        self._query_expander = query_expander
        self._cache = cache
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

        search_query = (
            query_result.rewrite.rewritten_query
            if query_result is not None
            else query
        )
        if query_result is not None and query_result.rewrite.sub_queries:
            retrieval = self._multi_retrieve(
                search_query,
                query_result.rewrite.sub_queries,
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
            )
        elif self._query_expander is not None and not isinstance(
            self._query_expander, IdentityExpander
        ):
            variants = self._query_expander.expand(search_query)
            retrieval = self._multi_retrieve(
                search_query,
                variants[1:] if len(variants) > 1 else [],
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
            )
        else:
            retrieval = self._retriever.search(
                search_query,
                k=k or self._config.k,
                filter=filter,
                use_lexical=self._resolved(use_lexical, self._config.use_lexical),
                use_rerank=self._resolved(use_rerank, self._config.use_rerank),
            )

        answer = self._reader.answer(query, retrieval.chunks)
        result = QAResult(
            query=query,
            query_result=query_result,
            retrieval=retrieval,
            answer=answer,
        )
        self._maybe_store(query, result, use_cache)
        return result

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
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
    }


__all__ = [
    "QAEngine",
    "QACache",
    "QAResult",
]
