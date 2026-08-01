"""The :class:`KnowledgeBase` aggregate root: documents + blocks + index.

SparkSage had a flat :class:`~sparksage.documents.DocumentStore` but no
*knowledge-base* concept -- no way to group documents, scope retrieval to one
tenant, or keep the vector index consistent with document edits.
:class:`KnowledgeBase` fills that gap. It is the aggregate root that owns:

* a set of :class:`~sparksage.documents.DocumentRecord`s,
* the :class:`~sparksage.schema.IdeaBlock`s produced from them,
* a dense :class:`~sparksage.embed.store.VectorStore` + a
  :class:`~sparksage.retrieve.lexical.BM25Retriever` lexical index, and
* a :class:`~sparksage.retrieve.Retriever` over both.

Crucially it owns the **consistency** between those layers -- the thing that
was missing before:

* adding a document embeds + indexes its blocks (``content_hash`` stamped),
* removing a document cascades to its block vectors + registry entries,
* ``content_hash`` change detection makes :meth:`update_document` an
  incremental re-index only when the body actually changed,
* :meth:`reindex` rebuilds the dense + lexical indexes from the live registry
  in one shot (the drift-recovery escape hatch).

Each block gets its ``kb_id`` stamped on ingest, so a
:class:`~sparksage.retrieve.RetrievalFilter` can scope retrieval to one KB --
the multi-tenant retrieval the analysis called for. The aggregate delegates
the actual search to its owned :class:`~sparksage.retrieve.Retriever`, so it
inherits hybrid recall + fusion + re-rank for free.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sparksage.documents.backends.memory import InMemoryDocumentStore
from sparksage.documents.models import DocumentRecord, content_hash_of
from sparksage.documents.store import DocumentStore
from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.store import InMemoryVectorStore, VectorStore
from sparksage.kb.models import KnowledgeBaseInfo
from sparksage.retrieve.lexical import BM25Retriever, LexicalRetriever, NullLexicalRetriever
from sparksage.retrieve.models import RetrievalFilter, RetrievalResult
from sparksage.retrieve.orchestrator import Retriever
from sparksage.retrieve.reranker import Reranker
from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Aggregate root: documents + blocks + a consistent vector index.

    Parameters
    ----------
    info:
        :class:`KnowledgeBaseInfo` metadata (id / name / language / ACL / ...).
    embedder:
        :class:`BlockEmbedder` used to vectorize blocks and queries.
    store:
        Optional dense :class:`VectorStore`. Defaults to an
        :class:`~sparksage.embed.store.InMemoryVectorStore` sized to the
        embedder's dimension.
    lexical:
        Optional :class:`LexicalRetriever`. Defaults to a
        :class:`~sparksage.retrieve.lexical.BM25Retriever`. Pass
        :class:`~sparksage.retrieve.lexical.NullLexicalRetriever` to disable
        the lexical leg.
    document_store:
        Optional :class:`DocumentStore` for the documents. Defaults to an
        :class:`~sparksage.documents.backends.memory.InMemoryDocumentStore`.
    reranker:
        Optional :class:`Reranker` forwarded to the owned :class:`Retriever`.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient
    >>> from sparksage.kb import KnowledgeBase, KnowledgeBaseInfo
    >>> kb = KnowledgeBase(
    ...     info=KnowledgeBaseInfo(name="docs"),
    ...     embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
    ... )
    >>> kb.add_blocks(blocks)            # doctest: +SKIP
    >>> result = kb.search("how to deploy", k=3)   # doctest: +SKIP
    """

    def __init__(
        self,
        info: KnowledgeBaseInfo,
        embedder: BlockEmbedder,
        *,
        store: VectorStore | None = None,
        lexical: LexicalRetriever | None = None,
        document_store: DocumentStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._info = info
        self._embedder = embedder
        self._store: VectorStore = store if store is not None else InMemoryVectorStore(
            embedder.dimension
        )
        self._lexical: LexicalRetriever = (
            lexical if lexical is not None else BM25Retriever()
        )
        self._document_store: DocumentStore = (
            document_store if document_store is not None else InMemoryDocumentStore()
        )
        self._registry: dict[str, IdeaBlock] = {}
        self._doc_blocks: dict[str, set[str]] = {}
        self._block_doc: dict[str, str] = {}
        # Doc ids owned by this KB (per-KB accounting over a possibly-shared store).
        self._doc_ids: set[str] = set()
        self._retriever = Retriever(
            self._registry,
            self._store,
            self._embedder,
            lexical=self._lexical,
            reranker=reranker,
        )

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #
    @property
    def info(self) -> KnowledgeBaseInfo:
        return self._info

    @property
    def kb_id(self) -> str:
        return self._info.kb_id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def embedder(self) -> BlockEmbedder:
        return self._embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def lexical(self) -> LexicalRetriever:
        return self._lexical

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def document_store(self) -> DocumentStore:
        return self._document_store

    def block_count(self) -> int:
        """Number of blocks currently in the registry."""
        return len(self._registry)

    def document_count(self) -> int:
        """Number of documents owned by this knowledge base.

        With a shared (multi-tenant) store this is the *per-KB* count -- only
        documents added through this KB are counted -- so the knowledge-base
        browse page stays accurate even when several KBs share one store.
        """
        return len(self._doc_ids)

    def owns_document(self, doc_id: str) -> bool:
        """Whether this KB owns ``doc_id`` (added via :meth:`add_document`)."""
        return str(doc_id) in self._doc_ids

    def blocks(self) -> list[IdeaBlock]:
        """Return a snapshot of all blocks in the registry."""
        return list(self._registry.values())

    def get_block(self, block_id: str) -> IdeaBlock | None:
        """Return the block for ``block_id`` (a copy is not taken)."""
        return self._registry.get(str(block_id))

    def blocks_for_document(self, doc_id: str) -> list[IdeaBlock]:
        """Return the blocks linked to ``doc_id``."""
        ids = self._doc_blocks.get(str(doc_id), set())
        out = [self._registry[bid] for bid in ids if bid in self._registry]
        return out

    # ------------------------------------------------------------------ #
    # block / document mutation (consistency is maintained here)
    # ------------------------------------------------------------------ #
    def add_blocks(
        self,
        blocks: Iterable[IdeaBlock],
        *,
        doc_id: str | None = None,
    ) -> list[IdeaBlock]:
        """Register, embed and index ``blocks`` (stamping ``kb_id`` on each).

        Newly added blocks overwrite any prior block with the same id (the
        vector store overwrites in place; the lexical index is rebuilt).
        Pass ``doc_id`` to link the blocks to a document so
        :meth:`remove_document` cascades to them.

        Returns the (possibly kb-id-stamped) list of added blocks.
        """
        added: list[IdeaBlock] = []
        new_vectors: dict[str, list[float]] = {}
        for block in blocks:
            if block.kb_id is None:
                block.kb_id = self.kb_id
            bid = str(block.id)
            self._registry[bid] = block
            if doc_id is not None:
                self._link_block(bid, str(doc_id))
            added.append(block)

        if added:
            self._embedder.embed_blocks(added)
            for b in added:
                if b.embedding is not None:
                    new_vectors[str(b.id)] = list(b.embedding)
            if new_vectors:
                self._store.add_many(new_vectors)
            self._rebuild_lexical()
        return added

    def remove_block(self, block_id: str) -> bool:
        """Remove ``block_id`` from the registry, vector store and lexical index.

        Returns whether a block was actually removed.
        """
        bid = str(block_id)
        if bid not in self._registry:
            return False
        del self._registry[bid]
        self._store.remove(bid)
        doc_id = self._block_doc.pop(bid, None)
        if doc_id is not None and doc_id in self._doc_blocks:
            self._doc_blocks[doc_id].discard(bid)
            if not self._doc_blocks[doc_id]:
                del self._doc_blocks[doc_id]
        self._rebuild_lexical()
        return True

    def add_document(
        self,
        record: DocumentRecord,
        blocks: Iterable[IdeaBlock] | None = None,
    ) -> DocumentRecord:
        """Store ``record`` and (optionally) index its ``blocks``.

        The document is saved first, then any blocks are added and linked to
        the document's ``doc_id``. ``content_hash`` change detection is the
        caller's responsibility for the *first* write; use
        :meth:`update_document` for hash-aware incremental re-indexing.
        """
        stored = self._document_store.save(record)
        self._doc_ids.add(str(stored.doc_id))
        if blocks is not None:
            self.add_blocks(blocks, doc_id=stored.doc_id)
        return stored

    def update_document(
        self,
        doc_id: str,
        record: DocumentRecord | None = None,
        blocks: Iterable[IdeaBlock] | None = None,
    ) -> DocumentRecord:
        """Hash-aware incremental update of a document and its blocks.

        If the (new) document body's ``content_hash`` matches the stored one
        *and* no ``blocks`` are supplied, this is a no-op -- the index already
        reflects the content, so we skip a needless re-embed. When the body
        changed (or blocks were supplied) the old linked blocks are removed and
        the new ones indexed, keeping the vector store consistent with the
        document (the consistency gap the analysis flagged).
        """
        doc_id = str(doc_id)
        if doc_id not in self._doc_ids:
            raise KeyError(f"document not found: {doc_id}")
        existing = self._document_store.get(doc_id)
        if existing is None:
            raise KeyError(f"document not found: {doc_id}")

        new_record = record if record is not None else existing
        body_changed = (new_record.content_hash or content_hash_of(
            new_record.body_markdown
        )) != (existing.content_hash)

        stored = self._document_store.save(new_record)
        if blocks is None and not body_changed:
            return stored

        for bid in list(self._doc_blocks.get(doc_id, set())):
            self.remove_block(bid)
        if blocks is not None:
            self.add_blocks(blocks, doc_id=doc_id)
        return stored

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document *and* cascade-remove its linked blocks.

        This is the index<->storage consistency guarantee: deleting a document
        also deletes its block vectors + registry entries, so the index can
        never serve orphaned chunks from a removed document. With a shared
        store only this KB's own documents are removable -- a ``doc_id`` owned
        by another KB is left untouched (returns ``False``).
        """
        doc_id = str(doc_id)
        if doc_id not in self._doc_ids:
            return False
        self._doc_ids.discard(doc_id)
        self._document_store.delete(doc_id)
        for bid in list(self._doc_blocks.get(doc_id, set())):
            self.remove_block(bid)
        return True

    def reindex(self) -> int:
        """Rebuild the dense + lexical indexes from the live registry.

        The drift-recovery escape hatch: re-embeds every block and rebuilds the
        lexical index from scratch, guaranteeing the indexes match the registry
        exactly. Returns the number of blocks re-indexed.
        """
        blocks = list(self._registry.values())
        self._store.clear()
        self._lexical.index([])
        if not blocks:
            return 0
        self._embedder.embed_blocks(blocks)
        vectors = self._embedder.vectors_for(blocks)
        self._store.add_many(vectors)
        if not isinstance(self._lexical, NullLexicalRetriever):
            self._lexical.index(blocks)
        return len(blocks)

    # ------------------------------------------------------------------ #
    # retrieval (delegates to the owned Retriever)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: RetrievalFilter | None = None,
        use_lexical: bool = True,
        use_rerank: bool = True,
    ) -> RetrievalResult:
        """Scope-aware retrieval over this KB.

        A :class:`RetrievalFilter` with ``kb_id`` set to *this* KB is implied
        (the registry only contains this KB's blocks anyway, but setting it
        makes the intent explicit and future-proofs cross-KB stores).
        """
        effective = filter if filter is not None else RetrievalFilter()
        if effective.kb_id is None:
            from dataclasses import replace

            effective = replace(effective, kb_id=self.kb_id)
        return self._retriever.search(
            query,
            k=k,
            filter=effective,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _link_block(self, block_id: str, doc_id: str) -> None:
        prev_doc = self._block_doc.get(block_id)
        if prev_doc is not None and prev_doc != doc_id:
            self._doc_blocks.setdefault(prev_doc, set()).discard(block_id)
        self._block_doc[block_id] = doc_id
        self._doc_blocks.setdefault(doc_id, set()).add(block_id)

    def _rebuild_lexical(self) -> None:
        if isinstance(self._lexical, NullLexicalRetriever):
            return
        self._lexical.index(list(self._registry.values()))


__all__ = ["KnowledgeBase"]
