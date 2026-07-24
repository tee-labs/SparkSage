"""Dense-vector embedding for IdeaBlocks.

Inject any :class:`EmbeddingClient` (a real :class:`OpenAIEmbeddingClient` in
production, or :class:`FakeEmbeddingClient` in tests) into a
:class:`BlockEmbedder` and call :meth:`~BlockEmbedder.embed_blocks` to fill the
``embedding`` field of each :class:`~sparksage.schema.IdeaBlock`.

Only :attr:`~sparksage.schema.IdeaBlock.embedding_text` is ever embedded.
"""

from sparksage.embed.client import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from sparksage.embed.indexer import BlockEmbedder

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_WORKERS",
    "BlockEmbedder",
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "OpenAIEmbeddingClient",
]
