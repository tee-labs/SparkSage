"""The :class:`IdeaBlock`: SparkSage's question-aligned structured chunk.

An IdeaBlock replaces the traditional "fixed-size text slice" with a small,
self-contained *knowledge unit* that is aligned to how users ask questions.

Design principles (borrowed & adapted from the Blockify methodology):

* **Question-answer alignment** -- every block carries the question it answers
  (``critical_question``) plus a verified, self-consistent answer
  (``trusted_answer``). This aligns the chunk to the query manifold, so dense
  vectors cluster tightly around user intent instead of around arbitrary text.
* **Single-field embedding** -- only ``trusted_answer`` is embedded by default,
  which kills the "the splitter cut my sentence in half" problem that wrecks
  naive chunking.
* **Rich, queryable metadata** -- ``tags`` / ``entities`` / ``keywords`` power
  filtering, permission scoping and hybrid (BM25 + dense) retrieval.
* **Provenance & lifecycle** -- every block knows where it came from
  (:class:`SourceRef`) and its dedup state (``status`` / ``parents``), so the
  corpus is auditable and the Distill pipeline can merge safely.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparksage.schema.entity import Entity
from sparksage.schema.enums import BlockStatus, Tag
from sparksage.schema.source import SourceRef

#: Recommended soft cap (in characters) for ``trusted_answer``. Concise,
#: self-contained answers are what make question-aligned chunks retrieve well.
RECOMMENDED_ANSWER_MAX = 500

#: Hard cap (in characters) for ``critical_question``.
QUESTION_MAX = 300

#: Delimiter used when concatenating fields into embedding text.
_EMBED_DELIM = "\n"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IdeaBlock(BaseModel):
    """A single question-aligned knowledge chunk -- the core SparkSage unit."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    # --- identity ----------------------------------------------------------
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Stable unique id of the block.",
    )

    # --- question-answer core (the innovation) -----------------------------
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short title/headline summarising what this block is about.",
    )
    critical_question: str = Field(
        ...,
        max_length=QUESTION_MAX,
        description="The single question this block exists to answer.",
    )
    trusted_answer: str = Field(
        ...,
        min_length=1,
        description=(
            "Verified, self-consistent answer (2-3 sentences, "
            f"<={RECOMMENDED_ANSWER_MAX} chars). Embedded as a single field."
        ),
    )

    # --- retrieval-oriented metadata ---------------------------------------
    tags: list[Tag] = Field(
        default_factory=list,
        description="Coarse semantic tags for filtering and policy.",
    )
    entities: list[Entity] = Field(
        default_factory=list,
        description="Named things this block references (entity graph).",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for BM25 / lexical recall boosting.",
    )

    # --- provenance --------------------------------------------------------
    source: SourceRef | None = Field(
        default=None,
        description="Where this block was extracted from (auditability).",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=16,
        description="ISO-639/BCP-47 language code of the answer text.",
    )
    author: str | None = Field(default=None, description="Who created/verified it.")
    version: int = Field(default=1, ge=1, description="Monotonic content version.")

    # --- lifecycle / distill bookkeeping -----------------------------------
    status: BlockStatus = Field(
        default=BlockStatus.DRAFT,
        description="Lifecycle status (used by the Distill dedup pipeline).",
    )
    parents: list[uuid.UUID] = Field(
        default_factory=list,
        description="UUIDs of blocks merged into this one (provenance chain).",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Merge/embedding confidence, if produced by Distill.",
    )
    embedding: list[float] | None = Field(
        default=None,
        repr=False,
        description="Optional dense vector for the answer (populated downstream).",
    )

    # --- timestamps --------------------------------------------------------
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # ------------------------------------------------------------------ #
    # validators
    # ------------------------------------------------------------------ #
    @field_validator("name", "critical_question", "trusted_answer")
    @classmethod
    def _strip_required_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("required text fields must not be empty/whitespace")
        return v

    @field_validator("trusted_answer")
    @classmethod
    def _enforce_answer_brevity(cls, v: str) -> str:
        if len(v) > RECOMMENDED_ANSWER_MAX:
            raise ValueError(
                f"trusted_answer is {len(v)} chars; keep it concise "
                f"(<={RECOMMENDED_ANSWER_MAX}). Split into another IdeaBlock instead."
            )
        return v

    @field_validator("critical_question")
    @classmethod
    def _encourage_question_form(cls, v: str) -> str:
        # Soft nudge: question-aligned blocks retrieve best as real questions.
        stripped = v.rstrip().rstrip(".!。")
        if not stripped.endswith(("?", "？")):
            raise ValueError(
                "critical_question should be phrased as a question (end with '?')."
            )
        return v

    @field_validator("keywords")
    @classmethod
    def _normalize_keywords(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for kw in v:
            kw = kw.strip()
            if kw and kw.lower() not in {k.lower() for k in out}:
                out.append(kw)
        return out

    @field_validator("tags")
    @classmethod
    def _dedupe_tags(cls, v: list[Tag]) -> list[Tag]:
        seen: list[Tag] = []
        for t in v:
            if t not in seen:
                seen.append(t)
        return seen

    # ------------------------------------------------------------------ #
    # computed helpers
    # ------------------------------------------------------------------ #
    @property
    def embedding_text(self) -> str:
        """Text used to build the dense vector: name + question + answer.

        Concatenating these three fields (the Distill convention) gives the
        embedder both *what* and *how it's asked*, producing tighter clusters
        than embedding raw prose. Only this string should be embedded.
        """
        return _EMBED_DELIM.join(
            [self.name, self.critical_question, self.trusted_answer]
        )

    @property
    def is_live(self) -> bool:
        """Whether this block should be returned by retrieval by default."""
        return self.status == BlockStatus.ACTIVE

    def touch(self) -> IdeaBlock:
        """Bump ``updated_at`` and ``version`` after a content edit."""
        self.updated_at = _utcnow()
        self.version += 1
        return self

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def to_xml(self) -> str:
        """Serialize to the human-readable IdeaBlock XML form.

        This mirrors the canonical IdeaBlock representation so blocks can be
        exchanged with / inspected by external tooling.
        """
        from xml.sax.saxutils import escape

        def _list(values: list[Any]) -> str:
            return ", ".join(str(v.value if hasattr(v, "value") else v) for v in values)

        parts = ["<ideablock>"]
        parts.append(f"  <name>{escape(self.name)}</name>")
        parts.append(f"  <critical_question>{escape(self.critical_question)}</critical_question>")
        parts.append(f"  <trusted_answer>{escape(self.trusted_answer)}</trusted_answer>")
        parts.append(f"  <tags>{escape(_list(self.tags))}</tags>")
        parts.append(f"  <keywords>{escape(_list(self.keywords))}</keywords>")
        for ent in self.entities:
            parts.append("  <entity>")
            parts.append(f"    <entity_name>{escape(ent.entity_name)}</entity_name>")
            parts.append(f"    <entity_type>{ent.entity_type.value}</entity_type>")
            if ent.aliases:
                parts.append(f"    <aliases>{escape(', '.join(ent.aliases))}</aliases>")
            parts.append("  </entity>")
        parts.append("</ideablock>")
        return "\n".join(parts)

    def to_searchable_dict(self) -> dict[str, Any]:
        """Flat dict optimized for indexing into a vector/search store.

        Embeddings/LSH/keyword indexers usually want denormalized, flat
        records; this drops nested pydantic noise and exposes the fields they
        actually consume.
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "critical_question": self.critical_question,
            "trusted_answer": self.trusted_answer,
            "embedding_text": self.embedding_text,
            "tags": [t.value for t in self.tags],
            "keywords": list(self.keywords),
            "entities": [e.entity_name for e in self.entities],
            "entity_types": [e.entity_type.value for e in self.entities],
            "language": self.language,
            "status": self.status.value,
            "version": self.version,
            "source_uri": self.source.uri if self.source else None,
            "parents": [str(p) for p in self.parents],
        }
