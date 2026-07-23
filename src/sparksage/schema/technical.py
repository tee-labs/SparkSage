"""TechnicalBlock: the order-sensitive variant of an IdeaBlock.

Most RAG tooling treats content as an unordered bag of facts. That breaks for
manuals, SOPs, runbooks and tutorials, where *sequence* is meaning -- a
``PREREQUISITE`` must precede the ``COMMAND`` it guards, and a ``WARNING`` is
useless if detached from the step it modifies.

A :class:`TechnicalBlock` keeps the question-aligned IdeaBlock core (so it
embeds and retrieves the same way) but layers in:

* **ordered, role-tagged sentences** (:class:`AnnotatedSentence`), and
* **Primary / Proceeding / Following** context windows that preserve the
  surrounding flow a block was extracted from.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparksage.schema.enums import SentenceRole
from sparksage.schema.ideablock import _EMBED_DELIM, IdeaBlock


class AnnotatedSentence(BaseModel):
    """A single sentence with a structural role inside a technical block."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="The sentence content.")
    role: SentenceRole = Field(
        default=SentenceRole.INFO,
        description="Structural role of the sentence (COMMAND, WARNING, ...).",
    )

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sentence text must not be empty")
        return v


class _ContextWindow(BaseModel):
    """Surrounding-flow context for an ordered block (Primary/Proc/Following)."""

    model_config = ConfigDict(extra="forbid")

    primary: str = Field(default="", description="The main section the block lives in.")
    proceeding: str = Field(
        default="", description="Text immediately preceding the block."
    )
    following: str = Field(
        default="", description="Text immediately following the block."
    )


class TechnicalBlock(IdeaBlock):
    """An order-sensitive IdeaBlock for manuals / SOPs / runbooks.

    Inherits the full question-aligned core, metadata, provenance and lifecycle
    of :class:`IdeaBlock`, so it interoperates with the same retrieval stack.
    """

    context: _ContextWindow = Field(
        default_factory=_ContextWindow,
        description="Primary/Proceeding/Following flow context.",
    )
    steps: list[AnnotatedSentence] = Field(
        ...,
        min_length=1,
        description="Ordered, role-tagged sentences that make up the procedure.",
    )

    @field_validator("steps")
    @classmethod
    def _clean_steps(cls, v: list[AnnotatedSentence]) -> list[AnnotatedSentence]:
        # Drop exact consecutive duplicates that arise from sloppy ingestion.
        cleaned: list[AnnotatedSentence] = []
        prev: str | None = None
        for s in v:
            if s.text == prev:
                continue
            cleaned.append(s)
            prev = s.text
        if not cleaned:
            raise ValueError("TechnicalBlock requires at least one step")
        return cleaned

    @property
    def embedding_text(self) -> str:
        """Embedding text enriched with ordered, role-tagged steps.

        Falls back to the base IdeaBlock concatenation when there are no steps,
        so plain prose blocks embed identically to :class:`IdeaBlock`.
        """
        if not self.steps:
            return super().embedding_text
        numbered = [
            f"{i + 1}. [{s.role.value}] {s.text}" for i, s in enumerate(self.steps)
        ]
        return _EMBED_DELIM.join([self.name, self.critical_question, *numbered])

    @property
    def commands(self) -> list[AnnotatedSentence]:
        """Just the actionable steps (role == COMMAND)."""
        return [s for s in self.steps if s.role == SentenceRole.COMMAND]

    @property
    def warnings(self) -> list[AnnotatedSentence]:
        """Just the safety warnings (role == WARNING)."""
        return [s for s in self.steps if s.role == SentenceRole.WARNING]

    def to_searchable_dict(self) -> dict[str, Any]:
        data = super().to_searchable_dict()
        data.update(
            {
                "steps": [
                    {"text": s.text, "role": s.role.value} for s in self.steps
                ],
                "context_primary": self.context.primary,
                "context_proceeding": self.context.proceeding,
                "context_following": self.context.following,
            }
        )
        return data
