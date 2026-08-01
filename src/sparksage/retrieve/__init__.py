"""Hybrid retrieval + rank fusion + re-ranking for RAG.

This is the retrieval-orchestration layer that consumes the vectors / keywords
the ingest side has been producing and returns ranked chunks ready to feed a
reader. It closes the "single-stage pure-vector" gap and finally wires up the
three IdeaBlock fields that were *designed but unconsumed*:

* ``keywords``  -> :class:`BM25Retriever` (lexical recall boosting).
* ``tags`` / ``entities`` / ``language`` / ``kb_id`` -> :class:`RetrievalFilter`
  (multi-tenant metadata scoping).
* ``source.locator`` -> :class:`RetrievedChunk.to_citation` (grounded
  citations, consumed by :mod:`sparksage.reader`).

Everything depends only on protocols (:class:`~sparksage.embed.store.VectorStore`,
:class:`~sparksage.embed.indexer.BlockEmbedder`, :class:`LexicalRetriever`,
:class:`Reranker`) -- so the core is fully unit-testable with
:class:`~sparksage.embed.FakeEmbeddingClient` and zero network calls.

Pipeline::

    query
        -> dense (VectorStore kNN over embedding_text)
        -> lexical (BM25Retriever over keywords + answer)   [optional]
        -> weighted reciprocal rank fusion                  [when >1 leg]
        -> RetrievalFilter scoping (tags / entities / ...)
        -> re-rank (LLMReranker)                             [optional]
        -> score floor + decayed-retry / top-1 fallback      [optional]
        -> top-k RetrievedChunk list

Example
-------
::

    from sparksage import BlockEmbedder, FakeEmbeddingClient, InMemoryVectorStore
    from sparksage.retrieve import BM25Retriever, Retriever

    store = InMemoryVectorStore(dimension=64)
    retriever = Retriever(
        registry=registry,
        store=store,
        embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
        lexical=BM25Retriever(),
    )
    retriever.index(blocks)
    result = retriever.search("how to deploy", k=5)
"""

from sparksage.retrieve.fusion import (
    DEFAULT_RRF_K,
    DEFAULT_TUNE_K_CANDIDATES,
    DEFAULT_TUNE_WEIGHT_CANDIDATES,
    reciprocal_rank_fusion,
    tune_rrf_k,
    tune_rrf_weights,
)
from sparksage.retrieve.grader import (
    DEFAULT_GRADE_TOP_K,
    DEFAULT_RELEVANCE,
    GraderEmptyResponseError,
    GraderError,
    GraderResponseParseError,
    LLMRetrievalGrader,
    RawRelevance,
    RelevanceResult,
    RetrievalGrader,
    coerce_relevance,
    grade_messages,
    grade_system_prompt,
    grade_user_prompt,
    parse_raw_relevance,
    parse_relevance_response,
)
from sparksage.retrieve.lexical import (
    DEFAULT_B,
    DEFAULT_K1,
    KEYWORD_WEIGHT,
    BM25Retriever,
    LexicalRetriever,
    NullLexicalRetriever,
    tokenize,
)
from sparksage.retrieve.models import (
    Citation,
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
)
from sparksage.retrieve.orchestrator import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_FETCH_FACTOR,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_MIN_FETCH,
    DEFAULT_SCORE_MIN_TOP1,
    DEFAULT_SCORE_RETRY_FACTOR,
    DEFAULT_SCORE_RETRY_FLOOR,
    RetrievalConfig,
    Retriever,
)
from sparksage.retrieve.reranker import IdentityReranker, LLMReranker, Reranker

__all__ = [
    "BM25Retriever",
    "Citation",
    "DEFAULT_B",
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_DENSE_WEIGHT",
    "DEFAULT_FETCH_FACTOR",
    "DEFAULT_GRADE_TOP_K",
    "DEFAULT_K1",
    "DEFAULT_LEXICAL_WEIGHT",
    "DEFAULT_MIN_FETCH",
    "DEFAULT_RELEVANCE",
    "DEFAULT_RRF_K",
    "DEFAULT_SCORE_MIN_TOP1",
    "DEFAULT_SCORE_RETRY_FACTOR",
    "DEFAULT_SCORE_RETRY_FLOOR",
    "DEFAULT_TUNE_K_CANDIDATES",
    "DEFAULT_TUNE_WEIGHT_CANDIDATES",
    "GraderEmptyResponseError",
    "GraderError",
    "GraderResponseParseError",
    "IdentityReranker",
    "KEYWORD_WEIGHT",
    "LLMReranker",
    "LLMRetrievalGrader",
    "LexicalRetriever",
    "NullLexicalRetriever",
    "RawRelevance",
    "Reranker",
    "RetrievalGrader",
    "RetrievedChunk",
    "RelevanceResult",
    "RetrievalConfig",
    "RetrievalFilter",
    "RetrievalResult",
    "Retriever",
    "coerce_relevance",
    "grade_messages",
    "grade_system_prompt",
    "grade_user_prompt",
    "parse_raw_relevance",
    "parse_relevance_response",
    "reciprocal_rank_fusion",
    "tokenize",
    "tune_rrf_k",
    "tune_rrf_weights",
]
