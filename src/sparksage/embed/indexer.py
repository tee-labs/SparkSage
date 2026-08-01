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

**Contextual Retrieval** (Anthropic, 2024): a block's
:attr:`~sparksage.schema.IdeaBlock.embedding_text` is a *synthetic* string
(name + question + answer) that drops the surrounding document context, so
anaphora-dense or short blocks retrieve worse than they should. Passing a short
``context_prefix`` (e.g. a document summary produced once via
:class:`~sparksage.documents.ExtractiveSummarizer`) prepends that context to
*every* block before embedding -- the same idea as Anthropic's Contextual
Retrieval / Late Chunking, at near-zero cost and without a long-context
embedding model. The prefix is an embed-time concern: it is *not* written back
to :attr:`~sparksage.schema.IdeaBlock.embedding_text`, and the query side
(:meth:`embed_texts`) never receives it -- only indexed blocks carry the extra
document anchor, which is exactly the intended asymmetry.
"""

from __future__ import annotations

import logging
import time

from sparksage.embed.client import EmbeddingClient
from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)

#: Sentinel distinguishing "argument not passed" from an explicit ``None``.
#:
#: ``context_prefix`` defaults to the constructor value; an explicit ``None`` on
#: a per-call basis means "no prefix for this call", while omitting it means
#: "use the constructor default". A bare ``None`` default cannot express both.
_UNSET: object = object()


class BlockEmbedder:
    """Fill :attr:`IdeaBlock.embedding` via a pluggable client.

    Parameters
    ----------
    client:
        Any :class:`EmbeddingClient` (e.g. :class:`OpenAIEmbeddingClient`,
        :class:`FakeEmbeddingClient`). Decouples vectorization from any specific
        SDK so the core stays unit-testable.
    context_prefix:
        Optional short text prepended to every block's
        :attr:`~sparksage.schema.IdeaBlock.embedding_text` before embedding --
        the Contextual Retrieval guard. Typically a one-time document summary
        (e.g. from :class:`~sparksage.documents.ExtractiveSummarizer`) that
        anchors each block to its source document, helping short /
        anaphora-dense blocks retrieve better. The prefix is **not** stored on
        the block and is **not** applied to raw query embedding
        (:meth:`embed_texts`) -- only indexed blocks carry the extra anchor.
        Per-call ``context_prefix`` overrides on :meth:`embed_blocks` /
        :meth:`vectors_for` take precedence (``None`` there disables it for that
        call). Default ``None`` (no prefix, the legacy behaviour).

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient
    >>> embedder = BlockEmbedder(FakeEmbeddingClient())
    >>> embedder.embed_blocks(blocks)          # fills .embedding in place
    [IdeaBlock(...), ...]

    Or without mutating the blocks (for a vector store)::

        vectors = embedder.vectors_for(blocks)

    With a document summary prepended for Contextual Retrieval::

        embedder = BlockEmbedder(
            FakeEmbeddingClient(), context_prefix="SparkSage deploy guide"
        )
        embedder.embed_blocks(blocks)   # each block anchored to the doc
    """

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        context_prefix: str | None = None,
    ) -> None:
        self._client = client
        self._context_prefix = context_prefix

    @property
    def client(self) -> EmbeddingClient:
        """The underlying :class:`EmbeddingClient` (mainly for inspection)."""
        return self._client

    @property
    def context_prefix(self) -> str | None:
        """The constructor-level Contextual Retrieval prefix, or ``None``."""
        return self._context_prefix

    @property
    def dimension(self) -> int:
        """The vector dimensionality this embedder produces."""
        return self._client.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings (low-level escape hatch).

        Returns one vector per input, in order. Honours the client's batching
        and concurrency internally. The Contextual Retrieval ``context_prefix``
        is deliberately **not** applied here -- this is the query side, which
        is embedded as-is so only indexed blocks carry the document anchor.
        """
        if not texts:
            return []
        t0 = time.perf_counter()
        vectors = self._client.embed_batch(list(texts))
        elapsed = time.perf_counter() - t0
        _logger.debug(
            "embed_texts: count=%d dim=%d elapsed=%.2fs",
            len(texts),
            self._client.dimension,
            elapsed,
        )
        return vectors

    def _resolve_prefix(self, context_prefix: object) -> str | None:
        if context_prefix is _UNSET:
            return self._context_prefix
        if context_prefix is None:
            return None
        if not isinstance(context_prefix, str):
            raise TypeError("context_prefix must be a str or None")
        return context_prefix

    def _embed_texts_with_prefix(
        self,
        blocks: list[IdeaBlock],
        prefix: str | None,
    ) -> list[list[float]]:
        texts = [
            f"{prefix}\n{b.embedding_text}" if prefix else b.embedding_text
            for b in blocks
        ]
        return self._client.embed_batch(texts)

    def embed_blocks(
        self,
        blocks: list[IdeaBlock],
        *,
        context_prefix: object = _UNSET,
    ) -> list[IdeaBlock]:
        """Fill ``.embedding`` on each block in place and return the list.

        Only :attr:`~sparksage.schema.IdeaBlock.embedding_text` is embedded
        (optionally prepended with the Contextual Retrieval ``context_prefix``).
        Blocks are mutated (not copied) for efficiency; the same list object is
        returned for fluent chaining.

        Parameters
        ----------
        context_prefix:
            Optional per-call override of the constructor's ``context_prefix``.
            Pass a ``str`` to anchor these blocks to a document summary for this
            call only, or ``None`` to disable the prefix for this call. Omit it
            (the default) to fall back to the constructor value.
        """
        if not blocks:
            return []
        prefix = self._resolve_prefix(context_prefix)
        _logger.debug(
            "embedding %d blocks (dim=%d, prefix=%s)",
            len(blocks),
            self._client.dimension,
            bool(prefix),
        )
        t0 = time.perf_counter()
        vectors = self._embed_texts_with_prefix(blocks, prefix)
        elapsed = time.perf_counter() - t0
        _logger.debug(
            "embed_blocks done: count=%d elapsed=%.2fs",
            len(blocks),
            elapsed,
        )
        for block, vec in zip(blocks, vectors, strict=True):
            block.embedding = list(vec)
        return blocks

    def vectors_for(
        self,
        blocks: list[IdeaBlock],
        *,
        context_prefix: object = _UNSET,
    ) -> dict[str, list[float]]:
        """Return ``{block_id: vector}`` *without* mutating the blocks.

        The contract a vector store or the future Distill pipeline consumes --
        vectors are keyed by the block's ``id`` (as a string) and deliberately
        kept off the block objects so large corpora stay memory-light.

        Parameters
        ----------
        context_prefix:
            Optional per-call override of the constructor's ``context_prefix``
            (see :meth:`embed_blocks`).
        """
        if not blocks:
            return {}
        prefix = self._resolve_prefix(context_prefix)
        vectors = self._embed_texts_with_prefix(blocks, prefix)
        return {
            str(b.id): list(v) for b, v in zip(blocks, vectors, strict=True)
        }
