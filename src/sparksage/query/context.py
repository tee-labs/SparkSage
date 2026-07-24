"""Conversation context for multi-turn query rewriting.

Multi-turn dialogue is where query rewriting earns its keep: a bare follow-up
like "那联通呢" / "what about Unicom?" only makes sense against the prior turns.
:class:`ConversationContext` is the first-class carrier for that history -- it is
passed to intent classifiers and rewriters so they can resolve anaphora
("那", "it", "the same") and inherit company / year / metric from earlier turns.

It is an immutable, pure-data value object: it carries no behaviour beyond
rendering itself into the text a prompt needs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Roles recognised by :meth:`ConversationContext.with_turn`.
USER = "user"
ASSISTANT = "assistant"
_VALID_ROLES = frozenset({USER, ASSISTANT})


@dataclass(frozen=True)
class ConversationTurn:
    """A single prior turn in the conversation.

    Attributes
    ----------
    role:
        ``"user"`` or ``"assistant"``.
    content:
        The turn text verbatim.
    """

    role: str
    content: str


@dataclass(frozen=True)
class ConversationContext:
    """Ordered, immutable history of a conversation.

    Build it incrementally with :meth:`with_turn` (returns a new context) or at
    once from ``(role, content)`` pairs via :meth:`from_pairs`. The empty context
    represents a first-turn (no history) query.
    """

    turns: tuple[ConversationTurn, ...] = ()

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[str, str]]
    ) -> ConversationContext:
        """Build a context from an iterable of ``(role, content)`` pairs."""
        ctx = cls()
        for role, content in pairs:
            ctx = ctx.with_turn(role, content)
        return ctx

    def with_turn(self, role: str, content: str) -> ConversationContext:
        """Return a new context with ``content`` appended as a ``role`` turn."""
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)!r}, got {role!r}"
            )
        return ConversationContext(
            self.turns + (ConversationTurn(role=role, content=content),)
        )

    def is_empty(self) -> bool:
        """True when there is no history (a first-turn query)."""
        return len(self.turns) == 0

    def as_text(self, *, max_turns: int | None = None) -> str:
        """Render the recent turns as ``role: content`` lines for a prompt.

        Returns the empty string when there is no history, so callers can always
        prepend the result. When ``max_turns`` is given, only the most recent
        that many turns are kept (older history is the least useful for
        coreference and just burns prompt tokens).
        """
        if not self.turns:
            return ""
        turns = self.turns if max_turns is None else self.turns[-max_turns:]
        return "\n".join(f"{t.role}: {t.content}" for t in turns)
