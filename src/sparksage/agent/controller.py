"""The agentic QA controller: the "brain" that decides the next action.

The controller is the single abstraction that makes the agentic loop
*paradigm-pluggable*: any callable that maps the current :class:`AgentState` to
the next :class:`AgentAction` is a controller. :class:`LLMAgentController`
implements a ReAct-style controller (thought -> action) on top of the existing
:class:`~sparksage.generator.LLMClient`; :class:`IdentityController` is the
no-op degenerate controller (always synthesize) so that
``AgenticQAEngine(IdentityController())`` collapses to the single-shot
``QAEngine`` baseline -- the same "off as a uniform protocol object" convention
the rest of the codebase uses (``IdentityReranker`` / ``IdentityRefiner`` /
``IdentityExpander``).

Everything depends only on the :class:`~sparksage.generator.LLMClient` protocol,
so the controller runs fully offline under
:class:`~sparksage.generator.FakeLLMClient`. Raw model output is coerced through
the :class:`~sparksage.agent.models.ActionType` enum via the lenient -> strict
schema (:mod:`sparksage.agent.schema`), keeping the enum the single source of
truth. On a bad response the LLM controller degrades to a synthesize action
rather than aborting a run (mirroring the reranker / expander fallback policy);
set ``strict=True`` to raise instead.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sparksage.agent.models import ActionType, AgentAction, AgentState
from sparksage.agent.prompts import (
    DEFAULT_EVIDENCE_ANSWER_CHARS,
    DEFAULT_EVIDENCE_TOP_K,
    agent_messages,
)
from sparksage.agent.schema import (
    CoercionError,
    parse_action_response,
)
from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient

_logger = logging.getLogger(__name__)


@runtime_checkable
class AgentController(Protocol):
    """Decide the next :class:`AgentAction` given the current :class:`AgentState`.

    A controller is consulted *after* the seed retrieval, so it always decides
    "retrieve another sub-query" vs "synthesize from what we have". Returning
    :attr:`ActionType.SYNTHESIZE` ends the loop.
    """

    def next_action(self, state: AgentState) -> AgentAction: ...


class AgentError(Exception):
    """Base class for controller errors."""


class ActionEmptyResponseError(AgentError):
    """The LLM returned an empty controller response."""


class ActionResponseParseError(AgentError):
    """The LLM controller response could not be parsed / coerced."""


class IdentityController:
    """The degenerate controller: always synthesize.

    With this controller :class:`~sparksage.agent.engine.AgenticQAEngine`
    performs exactly one (seed) retrieval and then synthesizes -- the same
    behaviour as the single-shot :class:`~sparksage.qa.QAEngine`. It is the
    cost-control / regression baseline, and the fallback target when an LLM
    controller call fails.
    """

    def next_action(self, state: AgentState) -> AgentAction:
        return AgentAction(
            action=ActionType.SYNTHESIZE,
            thought="identity controller: synthesize from seed evidence",
        )


class LLMAgentController:
    """ReAct-style controller backed by an :class:`LLMClient`.

    The model is shown the question, the multi-turn context, the steps already
    taken, and the evidence gathered so far, then asked to emit a JSON decision
    (``{"thought", "action", "query", "k"}``). The response is coerced through
    :class:`~sparksage.agent.models.ActionType` so the enum stays the single
    source of truth.

    On an empty / unparseable response the controller degrades to a
    :class:`IdentityController` decision (synthesize) rather than aborting a
    multi-step run -- so a single flaky call ends the loop gracefully with the
    evidence already gathered. Set ``strict=True`` to raise instead.

    Parameters
    ----------
    client:
        Any :class:`LLMClient`. Reused verbatim from the generator / reader.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.2``) for deliberate, repeatable planning.
    use_json_mode:
        Request JSON-mode structured output when supported.
    strict:
        If ``True``, raise on a bad response instead of falling back.
    evidence_top_k:
        How many evidence chunks to render back per turn (bounds prompt cost).
    evidence_answer_chars:
        Truncate each evidence ``trusted_answer`` to this many characters.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        use_json_mode: bool = True,
        strict: bool = False,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        evidence_answer_chars: int = DEFAULT_EVIDENCE_ANSWER_CHARS,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._strict = strict
        self._evidence_top_k = evidence_top_k
        self._evidence_answer_chars = evidence_answer_chars
        self._fallback = IdentityController()
        self.calls = 0
        self.fallbacks = 0

    @property
    def strict(self) -> bool:
        return self._strict

    def next_action(self, state: AgentState) -> AgentAction:
        self.calls += 1
        messages = agent_messages(
            state,
            evidence_top_k=self._evidence_top_k,
            evidence_answer_chars=self._evidence_answer_chars,
        )
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            return self._fallback_response("empty controller response")
        try:
            return parse_action_response(response_text, strict=self._strict)
        except CoercionError as exc:
            return self._fallback_response(f"parse error: {exc}")

    def _fallback_response(self, reason: str) -> AgentAction:
        if self._strict:
            if "empty" in reason:
                raise ActionEmptyResponseError(
                    "the LLM returned an empty controller response"
                )
            raise ActionResponseParseError(reason)
        self.fallbacks += 1
        _logger.warning("AgentController %s; falling back to synthesize", reason)
        return self._fallback.next_action(AgentState(question=""))


__all__ = [
    "ActionEmptyResponseError",
    "ActionResponseParseError",
    "AgentController",
    "AgentError",
    "IdentityController",
    "LLMAgentController",
]
