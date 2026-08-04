"""Shared parsing helpers for noisy LLM JSON output.

Every schema/parse module used to copy the same fence-stripping +
brace-matching ``extract_json``; this is the single copy they all import.
Callers pass their own error type / messages, and opt into array handling
(``allow_arrays``) or lenient repair (``lenient``) as needed.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json(
    text: str,
    error_type: type[Exception] = ValueError,
    *,
    empty_msg: str = "empty model response",
    invalid_msg: str = "model response was not valid JSON",
    allow_arrays: bool = False,
    lenient: bool = False,
) -> str:
    """Pull the JSON object/array out of a possibly-noisy model response.

    Handles plain JSON, JSON wrapped in ```json fences, and JSON embedded in
    surrounding prose (extracted via outermost brace/bracket matching). Raises
    ``error_type`` on an empty response; on an unparseable response raises
    ``error_type`` unless ``lenient``, in which case the cleaned text is
    returned so callers can attempt repair.
    """
    cleaned = text.strip()
    if not cleaned:
        raise error_type(empty_msg)

    cleaned = _FENCE_RE.sub("", cleaned).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    openers = (("[", "]"), ("{", "}")) if allow_arrays else (("{", "}"),)
    for open_ch, close_ch in openers:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    if lenient:
        return cleaned
    raise error_type(invalid_msg)
