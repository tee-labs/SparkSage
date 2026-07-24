"""LLM client abstraction for block generation.

The generation core depends *only* on the :class:`LLMClient` protocol, so it is
fully unit-testable with the deterministic :class:`FakeLLMClient`. A concrete
:class:`OpenAICompatibleClient` (backed by the ``openai`` SDK) is provided for
production use against any OpenAI-compatible Chat Completions endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Request JSON-mode structured output from providers that support it.
JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion interface the generator depends on.

    Any callable producing assistant message content from a list of OpenAI-style
    ``{"role", "content"}`` messages implements this -- a real HTTP client, a
    local model, or a deterministic fake for tests.
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant message content for ``messages``."""
        ...


class OpenAICompatibleClient:
    """LLM client backed by an OpenAI-compatible Chat Completions endpoint.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama's OpenAI shim, BigModel/GLM,
    and anything else that speaks the ``chat.completions`` protocol. Point it at
    a self-hosted / non-OpenAI endpoint via ``base_url``.

    By default requests are made in *streaming* mode (``stream=True``) and the
    streamed deltas are accumulated into the returned string. This is more
    robust for long generations (fewer timeouts) and is the behaviour the API
    layer enables. Disable it with ``stream=False`` or override it per call via
    ``complete(..., stream=False)``.

    The ``openai`` package is an *optional* dependency -- install it with
    ``pip install 'sparksage[llm]'``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float | None = None,
        stream: bool = True,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAICompatibleClient requires the 'openai' package. "
                "Install it with: pip install 'sparksage[llm]'"
            ) from exc
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, **client_kwargs
        )
        self._model = model
        self._stream = stream

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        stream = bool(kwargs.pop("stream", self._stream))
        request: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if response_format is not None:
            request["response_format"] = response_format
        request.update(kwargs)
        response = self._client.chat.completions.create(**request)
        if stream:
            parts: list[str] = []
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
            return "".join(parts)
        return response.choices[0].message.content or ""


@dataclass
class FakeLLMClient:
    """Deterministic, scriptable LLM client for tests and offline demos.

    Returns the preset ``responses`` in order, then replays the last one. The
    exact messages passed to each call are captured on ``last_messages`` so
    tests can assert on the prompt.
    """

    responses: list[str] = field(default_factory=list)
    index: int = 0
    last_messages: list[dict[str, str]] | None = None
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        self.last_messages = list(messages)
        self.calls.append(list(messages))
        if not self.responses:
            return ""
        if self.index < len(self.responses):
            text = self.responses[self.index]
            self.index += 1
            return text
        return self.responses[-1]
