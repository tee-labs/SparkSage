"""Data models for the retrieval layer.

These are the framework-agnostic contracts a retrieval orchestrator produces
and a reader (:mod:`sparksage.reader`) consumes. A :class:`RetrievedChunk`
carries the resolved :class:`~sparksage.schema.IdeaBlock` *plus* the dense /
lexical hit provenance so the reader can build grounded citations from
``source.locator`` (the field the ingest side has been filling but the query
side has not yet consumed).

Metadata filtering is expressed via :class:`RetrievalFilter`. Because the
:class:`~sparksage.embed.store.VectorStore` protocol is deliberately
text-agnostic (it indexes opaque ``block_id`` strings), filtering is applied
by the orchestrator against a block registry it holds -- this keeps the store
decoupled from embedding exactly as designed, while still enabling tag /
entity / language / KB scoping for multi-tenant retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparksage.embed.store import SearchHit
from sparksage.schema.enums import Tag
from sparksage.schema.ideablock import IdeaBlock


@dataclass(frozen=True)
class Citation:
    """A single grounded citation backing part of a generated answer.

    Attributes
    ----------
    block_id:
        The :class:`~sparksage.schema.IdeaBlock` id the claim came from.
    quote:
        The exact substring of the block's ``trusted_answer`` supporting the
        claim (for highlight rendering).
    uri:
        ``source.uri`` of the backing block -- where it was extracted from.
    locator:
        ``source.locator`` of the backing block -- position within the source
        (page, line range, anchor). This is the provenance field the schema
        has been carrying but no consumer has read until now.
    title:
        ``source.title`` of the backing block, for human-readable display.
    """

    block_id: str
    quote: str = ""
    uri: str | None = None
    locator: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval result: a block plus its score and provenance.

    Attributes
    ----------
    block:
        The resolved :class:`~sparksage.schema.IdeaBlock`.
    score:
        The (final, post-fusion / post-rerank) relevance score. Higher is
        better. Comparable across chunks within one result set.
    dense_score:
        The dense (cosine) similarity from the vector store, when applicable.
    lexical_score:
        The lexical (BM25) score from the keyword retriever, when applicable.
    rank:
        0-indexed position in the final ranked list.
    """

    block: IdeaBlock
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    rank: int = 0

    def to_citation(self, quote: str = "") -> Citation:
        """Build a :class:`Citation` from this chunk's block provenance."""
        src = self.block.source
        return Citation(
            block_id=str(self.block.id),
            quote=quote or self.block.trusted_answer,
            uri=src.uri if src else None,
            locator=src.locator if src else None,
            title=src.title if src else None,
        )


@dataclass
class RetrievalFilter:
    """Metadata scoping for retrieval (multi-tenant / permission / language).

    A block passes the filter when *every* set clause is satisfied:

    * ``tags``      -- the block carries at least one of these tags (ANY).
    * ``entities``  -- the block references at least one of these entity names
      (matched case-insensitively against ``entity_name`` and ``aliases``).
    * ``languages`` -- the block's ``language`` is one of these.
    * ``block_ids`` -- the block's id (as a string) is in this set; the
      coarsest possible scope (e.g. restrict to one document's blocks).
    * ``kb_id``     -- the block's ``kb_id`` equals this (see :mod:`sparksage.kb`).

    An empty filter (all-``None``) matches every block.
    """

    tags: set[Tag] | None = None
    entities: set[str] | None = None
    languages: set[str] | None = None
    block_ids: set[str] | None = None
    kb_id: str | None = None

    def matches(self, block: IdeaBlock) -> bool:
        """Return whether ``block`` satisfies this filter."""
        if self.tags is not None and not (set(block.tags) & self.tags):
            return False
        if self.entities is not None:
            names: set[str] = set()
            for ent in block.entities:
                names.add(ent.entity_name.lower())
                names.update(a.lower() for a in ent.aliases)
            if not (names & {e.lower() for e in self.entities}):
                return False
        if self.languages is not None and block.language not in self.languages:
            return False
        if self.block_ids is not None and str(block.id) not in self.block_ids:
            return False
        if self.kb_id is not None and getattr(block, "kb_id", None) != self.kb_id:
            return False
        return True

    @property
    def is_empty(self) -> bool:
        """Whether this filter imposes no constraints."""
        return not (
            self.tags
            or self.entities
            or self.languages
            or self.block_ids
            or self.kb_id
        )


@dataclass
class RetrievalResult:
    """The full outcome of one retrieval call.

    Attributes
    ----------
    query:
        The (possibly rewritten) query that was检索'd, for provenance.
    chunks:
        The final ranked :class:`RetrievedChunk` list (best first), after
        fusion and reranking.
    dense_hits:
        Raw dense :class:`~sparksage.embed.store.SearchHit` list (pre-fusion).
    lexical_hits:
        Raw lexical :class:`~sparksage.embed.store.SearchHit` list (pre-fusion),
        empty when lexical retrieval was disabled / unavailable.
    fused:
        Whether rank fusion (RRF) was applied.
    reranked:
        Whether a reranker reordered the candidates.
    filtered_out:
        How many candidates were dropped by :class:`RetrievalFilter` scoping.
    """

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    dense_hits: list[SearchHit] = field(default_factory=list)
    lexical_hits: list[SearchHit] = field(default_factory=list)
    fused: bool = False
    reranked: bool = False
    filtered_out: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether retrieval returned no usable chunks."""
        return not self.chunks

    @property
    def top_score(self) -> float:
        """Score of the best chunk (``0.0`` when empty)."""
        return self.chunks[0].score if self.chunks else 0.0


__all__ = [
    "Citation",
    "RetrievedChunk",
    "RetrievalFilter",
    "RetrievalResult",
]
