"""KnowledgeBase: the multi-tenant aggregate root for RAG.

SparkSage had a flat document store but no *knowledge-base* concept -- no way
to group documents, scope retrieval to one tenant, or keep the vector index
consistent with document edits. :class:`KnowledgeBase` fills that gap. It is
the aggregate root owning documents + their IdeaBlocks + a consistent dense +
lexical index, with:

* multi-document grouping and KB-scoped retrieval (``kb_id`` stamping +
  :class:`~sparksage.retrieve.RetrievalFilter`),
* index <--> storage **consistency** -- document delete cascades to block
  vectors; document update re-indexes only when ``content_hash`` changed,
* :meth:`~sparksage.kb.KnowledgeBase.reindex` -- the drift-recovery escape
  hatch.

A :class:`KnowledgeBaseStore` is the multi-tenant registry for the
serializable :class:`KnowledgeBaseInfo` metadata; the live vector index +
block registry are runtime state owned by the aggregate.

Example
-------
::

    from sparksage import BlockEmbedder, FakeEmbeddingClient
    from sparksage.kb import KnowledgeBase, KnowledgeBaseInfo

    kb = KnowledgeBase(
        info=KnowledgeBaseInfo(name="ops-docs"),
        embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
    )
    kb.add_blocks(blocks)
    result = kb.search("how to deploy", k=3)
"""

from sparksage.kb.knowledge_base import KnowledgeBase
from sparksage.kb.models import KnowledgeBaseInfo
from sparksage.kb.store import InMemoryKnowledgeBaseStore, KnowledgeBaseStore

__all__ = [
    "InMemoryKnowledgeBaseStore",
    "KnowledgeBase",
    "KnowledgeBaseInfo",
    "KnowledgeBaseStore",
]
