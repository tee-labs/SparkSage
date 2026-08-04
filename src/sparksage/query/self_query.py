"""Self-query parsing: turn natural language into ``(query, RetrievalFilter)``.

Users rarely phrase filtered questions the way a retrieval system needs them:
``"2025 年 6 月华东区销售数据"`` is really ``query="销售数据"`` scoped by
``{year: 2025, month: 6, region: 华东}``. A :class:`SelfQueryParser` closes that
gap by asking an LLM to split a question into a clean search query plus a
:class:`~sparksage.retrieve.RetrievalFilter` (tags / entities / languages),
which the retriever then applies as a post-filter.

The whole :class:`~sparksage.retrieve.RetrievalFilter` surface was already in
place but could only be built by hand -- this is the missing LLM front-end that
produces it from free text. It is the natural, low-cost next step on top of the
existing metadata-filtering infrastructure (zero orchestrator change: call the
parser, pass its ``filter`` into :meth:`~sparksage.retrieve.Retriever.search`).

The core depends only on the :class:`SelfQueryParser` protocol and reuses the
existing :class:`~sparksage.generator.LLMClient` (no new LLM abstraction), so it
is fully unit-testable with :class:`~sparksage.generator.FakeLLMClient`. The
lenient -> strict coercion follows the established
:mod:`sparksage.query.schema` pattern: raw model JSON is parsed into a lenient
:class:`RawSelfQuery` (``extra="ignore"``), then coerced through the closed
:class:`~sparksage.schema.enums.Tag` vocabulary into a strict
:class:`SelfQueryResult` -- so the controlled vocabularies stay the single
source of truth and a bad model call degrades to the identity parser (original
query + empty filter), never aborts the pipeline.

Two implementations ship:

- :class:`LLMSelfQueryParser`: the default -- asks the model for a JSON
  ``{query, tags, entities, languages}`` (tag values are read live from the
  :class:`~sparksage.schema.enums.Tag` enum, so extending it widens what the
  model may emit with no prompt edit).
- :class:`IdentitySelfQueryParser`: returns the query unchanged with an empty
  filter, so self-query is always configurable as "off" without branching.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.llmutil import extract_json as _extract_json
from sparksage.query.schema import CoercionError
from sparksage.retrieve.models import RetrievalFilter
from sparksage.schema.enums import Tag

_logger = logging.getLogger(__name__)


class SelfQueryError(RuntimeError):
    """Base error for self-query parsing."""


class SelfQueryEmptyResponseError(SelfQueryError):
    """The LLM returned no content."""


class SelfQueryResponseParseError(SelfQueryError):
    """The model response could not be parsed as the expected JSON."""


@dataclass
class SelfQueryResult:
    """The outcome of self-query parsing: a search query + a metadata filter.

    Attributes
    ----------
    query:
        The cleaned search query (structural / filtering terms stripped). Falls
        back to the original query when the model produced nothing usable.
    filter:
        A :class:`~sparksage.retrieve.RetrievalFilter` built from the extracted
        metadata. An empty filter (all-``None``) means "no scoping".
    """

    query: str
    filter: RetrievalFilter


@runtime_checkable
class SelfQueryParser(Protocol):
    """Parse one query into a :class:`SelfQueryResult` (query + filter)."""

    def parse(self, query: str) -> SelfQueryResult:
        """Return the :class:`SelfQueryResult` for ``query``."""
        ...


class IdentitySelfQueryParser:
    """A no-op parser: returns the query verbatim with an empty filter.

    Lets callers treat "self-query disabled" uniformly as a
    :class:`SelfQueryParser` rather than branching on ``None``.
    """

    def parse(self, query: str) -> SelfQueryResult:
        return SelfQueryResult(query=str(query), filter=RetrievalFilter())

    def __repr__(self) -> str:
        return "IdentitySelfQueryParser()"


# --------------------------------------------------------------------------- #
# Lenient model (what the LLM emits)
# --------------------------------------------------------------------------- #
class RawSelfQuery(BaseModel):
    """Lenient self-query decomposition as emitted by an LLM (extras ignored)."""

    model_config = ConfigDict(extra="ignore")

    query: str = ""
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# JSON extraction (from possibly-noisy model responses)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles plain JSON, JSON wrapped in ```json fences, and JSON embedded in
    surrounding prose (extracted via outermost brace matching). Raises
    :class:`CoercionError` on an empty or unparseable response.
    """
    return _extract_json(text, error_type=CoercionError)


def parse_raw_self_query(data: object) -> RawSelfQuery:
    """Validate decoded JSON into :class:`RawSelfQuery`."""
    if not isinstance(data, dict):
        raise CoercionError(
            f"expected a JSON object, got {type(data).__name__}"
        )
    try:
        return RawSelfQuery.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_self_query_response(text: str) -> RawSelfQuery:
    """Extract JSON from a raw model response and parse it into :class:`RawSelfQuery`."""
    return parse_raw_self_query(json.loads(extract_json(text)))


# --------------------------------------------------------------------------- #
# Coercion (lenient -> strict)
# --------------------------------------------------------------------------- #
def _tag_lookup(available_tags: Iterable[Tag]) -> dict[str, Tag]:
    """Build a case-insensitive ``value -> Tag`` index over ``available_tags``."""
    out: dict[str, Tag] = {}
    for t in available_tags:
        out[t.value.lower()] = t
    return out


def _clean_strings(items: list[str]) -> list[str]:
    """De-duplicate / strip a list of strings, dropping blanks (order-preserving)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def coerce_self_query(
    raw: RawSelfQuery,
    original_query: str,
    *,
    available_tags: Iterable[Tag] | None = None,
    strict_tags: bool = False,
) -> SelfQueryResult:
    """Normalize a :class:`RawSelfQuery` into a strict :class:`SelfQueryResult`.

    * ``query`` falls back to ``original_query`` when the model produced an empty
      one (a self-query that drops the query is never an improvement).
    * ``tags`` are mapped case-insensitively to :class:`~sparksage.schema.enums.Tag`
      members within ``available_tags`` (default: the whole enum). Foreign /
      unknown tags are silently dropped unless ``strict_tags`` raises on them.
    * ``entities`` / ``languages`` are free-form (matching the
      :class:`~sparksage.retrieve.RetrievalFilter` surface) and de-duplicated.
    """
    if not isinstance(raw, RawSelfQuery):
        raise CoercionError("expected a RawSelfQuery")

    lookup = _tag_lookup(available_tags if available_tags is not None else Tag)
    tags: set[Tag] = set()
    for raw_tag in raw.tags:
        key = (raw_tag or "").strip().lower()
        if not key:
            continue
        member = lookup.get(key)
        if member is not None:
            tags.add(member)
        elif strict_tags:
            raise CoercionError(f"unknown tag: {raw_tag!r}")

    entities = _clean_strings(raw.entities)
    languages = _clean_strings(raw.languages)

    flt = RetrievalFilter(
        tags=tags or None,
        entities=set(entities) or None,
        languages=set(languages) or None,
    )

    query = raw.query.strip()
    if not query:
        query = original_query.strip()

    return SelfQueryResult(query=query, filter=flt)


# --------------------------------------------------------------------------- #
# LLM-based parsing (the default)
# --------------------------------------------------------------------------- #
class LLMSelfQueryParser:
    """Self-query parser backed by an :class:`LLMClient`.

    The model is asked to split the question into a clean search query plus a
    metadata filter (tags / entities / languages). Tag values are read live from
    the :class:`~sparksage.schema.enums.Tag` enum into the prompt, so extending
    the enum widens what the model may emit with no prompt edit. The output is
    coerced leniently (handles prose-wrapped JSON, foreign tags, missing fields)
    and *always* falls back to the identity parser on an unparseable response --
    so a bad model call degrades to "no filtering", never aborts retrieval.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Reused verbatim from the rewriter / generator.
    model:
        Model name forwarded to the client (ignored by fakes). Self-query needs
        only a lightweight model -- set this to control cost.
    temperature:
        Low (default ``0.0``) for deterministic structured extraction.
    use_json_mode:
        Request JSON-mode structured output when supported.
    available_tags:
        The tag vocabulary the model may emit, as an iterable of
        :class:`~sparksage.schema.enums.Tag` members (default: every ``Tag``).
        Tags the model emits that are outside this set are dropped.
    strict_tags:
        If ``False`` (default), foreign tags are silently dropped. If ``True``,
        a foreign tag raises :class:`SelfQueryResponseParseError`.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        use_json_mode: bool = True,
        available_tags: Iterable[Tag] | None = None,
        strict_tags: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._available_tags = list(available_tags) if available_tags is not None else list(Tag)
        self._strict_tags = strict_tags
        self.fallbacks = 0

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def available_tags(self) -> list[Tag]:
        return list(self._available_tags)

    def parse(self, query: str) -> SelfQueryResult:
        original = str(query).strip()
        if not original:
            return SelfQueryResult(query="", filter=RetrievalFilter())

        tag_values = ", ".join(t.value for t in self._available_tags) or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a self-query decomposition module for metadata-aware "
                    "retrieval. Given a user question, split it into (a) a clean "
                    "search query with the structural / filtering terms removed, and "
                    "(b) a metadata filter. Only extract a filter value when the "
                    "question explicitly states it. Tags MUST come from this closed "
                    "vocabulary: "
                    f"[{tag_values}]. Entities are proper nouns (people, products, "
                    "organizations). Languages are ISO 639-1 codes (e.g. en, zh). "
                    "Respond with ONLY a JSON object: "
                    '{"query": "...", "tags": ["..."], "entities": ["..."], '
                    '"languages": ["..."]}. Omit empty arrays. No commentary.'
                ),
            },
            {
                "role": "user",
                "content": f"Question: {original}",
            },
        ]
        try:
            raw_text = self._client.complete(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
            )
            if not raw_text or not raw_text.strip():
                raise SelfQueryEmptyResponseError("the LLM returned an empty response")
            raw = parse_self_query_response(raw_text)
            return coerce_self_query(
                raw,
                original,
                available_tags=self._available_tags,
                strict_tags=self._strict_tags,
            )
        except (CoercionError, SelfQueryError, json.JSONDecodeError) as exc:
            self.fallbacks += 1
            _logger.warning(
                "LLMSelfQueryParser parse failure (%s); identity fallback", exc
            )
            return SelfQueryResult(query=original, filter=RetrievalFilter())

    def __repr__(self) -> str:
        return f"LLMSelfQueryParser(model={self._model!r})"


__all__ = [
    "IdentitySelfQueryParser",
    "LLMSelfQueryParser",
    "RawSelfQuery",
    "SelfQueryEmptyResponseError",
    "SelfQueryError",
    "SelfQueryParser",
    "SelfQueryResponseParseError",
    "SelfQueryResult",
    "coerce_self_query",
    "extract_json",
    "parse_raw_self_query",
    "parse_self_query_response",
]
