"""Dense-vector embedding for IdeaBlocks.

Inject any :class:`EmbeddingClient` (a real :class:`OpenAIEmbeddingClient` in
production, or :class:`FakeEmbeddingClient` in tests) into a
:class:`BlockEmbedder` and call :meth:`~BlockEmbedder.embed_blocks` to fill the
``embedding`` field of each :class:`~sparksage.schema.IdeaBlock`.

Only :attr:`~sparksage.schema.IdeaBlock.embedding_text` is ever embedded.

The produced vectors feed retrieval:

* :class:`InMemoryVectorStore` is a dependency-free kNN store -- add the
  ``{block_id: vector}`` mapping from :meth:`~BlockEmbedder.vectors_for` and
  :meth:`~InMemoryVectorStore.search` returns the most similar block ids.
* :func:`find_similar_pairs` is the all-pairs counterpart -- given the same
  ``{block_id: vector}`` mapping it returns the near-duplicate pairs (cosine
  >= threshold), the first step of the planned Distill de-dup pipeline.
* :func:`save_store` / :func:`load_store` persist a store to JSON so embeddings
  survive across restarts.
"""

from sparksage.embed.backends import (
    DEFAULT_CHROMA_SPACE,
    ChromaVectorStore,
    FaissVectorStore,
    PgvectorVectorStore,
)
from sparksage.embed.client import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.persist import (
    STORE_FORMAT,
    STORE_VERSION,
    load_store,
    save_store,
)
from sparksage.embed.similarity import (
    CandidateReducer,
    SimilarityPair,
    find_similar_pairs,
)
from sparksage.embed.store import (
    InMemoryVectorStore,
    SearchHit,
    VectorStore,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHROMA_SPACE",
    "DEFAULT_MAX_WORKERS",
    "STORE_FORMAT",
    "STORE_VERSION",
    "BlockEmbedder",
    "CandidateReducer",
    "ChromaVectorStore",
    "EmbeddingClient",
    "FaissVectorStore",
    "FakeEmbeddingClient",
    "InMemoryVectorStore",
    "OpenAIEmbeddingClient",
    "PgvectorVectorStore",
    "SearchHit",
    "SimilarityPair",
    "VectorStore",
    "find_similar_pairs",
    "load_store",
    "save_store",
]
