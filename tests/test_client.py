"""Tests for :class:`OpenAICompatibleClient` streaming behaviour.

The ``openai`` SDK is mocked out via ``sys.modules`` so these tests run fully
offline (no network, no real key) and assert on the *request shape* (whether
``stream=True`` was sent) and on delta accumulation.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeCreate:
    """Stand-in for ``client.chat.completions.create``."""

    def __init__(self, stream_chunks, nonstream_message):
        self.stream_chunks = stream_chunks
        self.nonstream_message = nonstream_message
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.stream_chunks)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.nonstream_message))
            ]
        )


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream_chunks=None,
    nonstream_message: str = "hello",
):
    create = _FakeCreate(stream_chunks or [], nonstream_message)
    module = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )

    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return create, _OpenAI


def _chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


_MSG = [{"role": "user", "content": "hi"}]


class TestStreaming:
    def test_stream_default_accumulates_deltas(self, monkeypatch):
        chunks = [_chunk("Hel"), _chunk("lo"), _chunk("!"), SimpleNamespace(choices=[])]
        create, _ = _install_fake_openai(monkeypatch, stream_chunks=chunks)
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k")
        result = client.complete(_MSG)

        assert result == "Hello!"
        assert create.calls[-1]["stream"] is True

    def test_stream_skips_none_delta_content(self, monkeypatch):
        chunks = [_chunk("AB"), _chunk(None), _chunk("CD")]
        create, _ = _install_fake_openai(monkeypatch, stream_chunks=chunks)
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k")
        assert client.complete(_MSG) == "ABCD"
        assert create.calls[-1]["stream"] is True

    def test_stream_off_returns_full_message(self, monkeypatch):
        create, _ = _install_fake_openai(monkeypatch, nonstream_message="full reply")
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k", stream=False)
        assert client.complete(_MSG) == "full reply"
        assert create.calls[-1]["stream"] is False

    def test_per_call_stream_override_beats_constructor(self, monkeypatch):
        chunks = [_chunk("XY")]
        create, _ = _install_fake_openai(
            monkeypatch, stream_chunks=chunks, nonstream_message="nonstream"
        )
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k", stream=False)
        assert client.complete(_MSG, stream=True) == "XY"
        assert create.calls[-1]["stream"] is True

    def test_per_call_disable_override(self, monkeypatch):
        create, _ = _install_fake_openai(monkeypatch, nonstream_message="plain")
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k")
        assert client.complete(_MSG, stream=False) == "plain"
        assert create.calls[-1]["stream"] is False

    def test_response_format_forwarded(self, monkeypatch):
        create, _ = _install_fake_openapi_with_stream(monkeypatch)
        from sparksage.generator.client import JSON_RESPONSE_FORMAT, OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k")
        client.complete(_MSG, response_format=JSON_RESPONSE_FORMAT)
        assert create.calls[-1]["response_format"] == {"type": "json_object"}


def _install_fake_openapi_with_stream(monkeypatch):
    return _install_fake_openai(
        monkeypatch, stream_chunks=[_chunk("ok")], nonstream_message="ok"
    )


class TestWiring:
    def test_base_url_and_api_key_forwarded_to_sdk(self, monkeypatch):
        _, OpenAI = _install_fake_openai(monkeypatch, nonstream_message="x")
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            base_url="https://my.host/v1", api_key="secret", model="glm-4"
        )
        assert client._client.init_kwargs["base_url"] == "https://my.host/v1"
        assert client._client.init_kwargs["api_key"] == "secret"

    def test_model_default_used_when_call_omits_it(self, monkeypatch):
        create, _ = _install_fake_openai(monkeypatch, stream_chunks=[_chunk("z")])
        from sparksage.generator.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="k", model="custom-model")
        client.complete(_MSG)
        assert create.calls[-1]["model"] == "custom-model"

    def test_missing_openai_raises_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        from sparksage.generator.client import OpenAICompatibleClient

        with pytest.raises(ImportError, match="sparksage\\[llm\\]"):
            OpenAICompatibleClient(api_key="k")
