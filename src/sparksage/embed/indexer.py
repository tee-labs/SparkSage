"""IdeaBlock vectorization orchestrator: blocks -> vectors.

:class:`BlockEmbedder` is the embedding counterpart of
:class:`~sparksage.generator.IdeaBlockGenerator`: it takes a list of
:class:`~sparksage.schema.IdeaBlock`s and fills their ``embedding`` field via a
pluggable :class:`~sparksage.embed.client.EmbeddingClient`. The text that gets
embedded is :attr:`~sparksage.schema.IdeaBlock.embedding_text` -- the *only*
text that should be embedded.

Design mirrors the rest of SparkSage: the embedder depends only on the
:class:`~sparksage.embed.client.EmbeddingClient` protocol, so it is deterministic
and fully unit-testable with
:class:`~sparksage.embed.client.FakeEmbeddingClient` and zero external
dependencies.

For large corpora where holding vectors on every block is impractical, use
:meth:`BlockEmbedder.vectors_for` to obtain a ``{block_id: vector}`` mapping
without mutating the blocks -- the contract a vector store / the future Distill
pipeline consumes.
"""

from __future__ import annotations

from sparksage.embed.client import EmbeddingClient
from sparksage.schema.ideablock import IdeaBlock


class BlockEmbedder:
    """Fill :attr:`IdeaBlock.embedding` via a pluggable client.

    Parameters
    ----------
    client:
        Any :class:`EmbeddingClient` (e.g. :class:`OpenAIEmbeddingClient`,
        :class:`FakeEmbeddingClient`). Decouples vectorization from any specific
        SDK so the core stays unit-testable.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient
    >>> embedder = BlockEmbedder(FakeEmbeddingClient())
    >>> embedder.embed_blocks(blocks)          # fills .embedding in place
    [IdeaBlock(...), ...]

    Or without mutating the blocks (for a vector store)::

        vectors = embedder.vectors_for(blocks)
    """

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    @property
    def client(self) -> EmbeddingClient:
        """The underlying :class:`EmbeddingClient` (mainly for inspection)."""
        return self._client

    @property
    def dimension(self) -> int:
        """The vector dimensionality this embedder produces."""
        return self._client.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings (low-level escape hatch).

        Returns one vector per input, in order. Honours the client's batching
        and concurrency internally.
        """
        if not texts:
            return []
        return self._client.embed_batch(list(texts))

    def embed_blocks(self, blocks: list[IdeaBlock]) -> list[IdeaBlock]:
        """Fill ``.embedding`` on each block in place and return the list.

        Only :attr:`~sparksage.schema.IdeaBlock.embedding_text` is embedded.
        Blocks are mutated (not copied) for efficiency; the same list object is
        returned for fluent chaining.
        """
        if not blocks:
            return []
        texts = [b.embedding_text for b in blocks]
        vectors = self._client.embed_batch(texts)
        for block, vec in zip(blocks, vectors, strict=True):
            block.embedding = list(vec)
        return blocks

    def vectors_for(self, blocks: list[IdeaBlock]) -> dict[str, list[float]]:
        """Return ``{block_id: vector}`` *without* mutating the blocks.

        The contract a vector store or the future Distill pipeline consumes --
        vectors are keyed by the block's ``id`` (as a string) and deliberately
        kept off the block objects so large corpora stay memory-light.
        """
        if not blocks:
            return {}
        texts = [b.embedding_text for b in blocks]
        vectors = self._client.embed_batch(texts)
        return {
            str(b.id): list(v) for b, v in zip(blocks, vectors, strict=True)
        }
