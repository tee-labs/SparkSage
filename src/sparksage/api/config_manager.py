"""Read / write the ``.env`` configuration file behind the API.

The WEB UI exposes the same ``.env``-based configuration the service reads at
startup so an operator can edit it from the browser. This module is the pure-
stdlib bridge between that UI and :mod:`sparksage.config`: it reads the current
effective values, masks secrets when echoing them back, and writes a patch of
new values back to the file.

Design constraints (matching :mod:`sparksage.config`):

* **Pure stdlib** -- no third-party env library. Reuses
  :func:`sparksage.config.parse_env_file` so the parsing rules stay in one
  place.
* **Real env vars win** -- :func:`read_config` reports the *effective* value of
  each known key (a real environment variable takes priority over the file),
  exactly like :func:`sparksage.config.load_dotenv` applies them. So the UI
  always shows what the running service actually uses.
* **Secrets are masked** -- any key ending in ``_API_KEY`` is returned as the
  sentinel ``"****"`` (never the real value) so the UI can render a password
  field without leaking the key.
* **Writes are a patch, not a rewrite** -- :func:`write_config` updates only
  the supplied keys, preserving comments, ordering and unknown keys in the rest
  of the file. Keys absent from the file are appended.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from sparksage.config import DEFAULT_ENV_FILENAME

__all__ = [
    "KNOWN_CONFIG_KEYS",
    "SENSITIVE_SUFFIX",
    "ConfigError",
    "mask_value",
    "read_config",
    "write_config",
]

#: Suffix identifying a secret. Any key ending with this is masked on read.
SENSITIVE_SUFFIX = "_API_KEY"

#: The set of SparkSage-relevant keys the UI knows about. The reader always
#: reports all of these (empty string when unset) so the form has a stable
#: shape; writes are still accepted for any ``*_API_KEY`` / ``SPARKSAGE_*`` /
#: ``OPENAI_*`` key so the UI can grow without a code change here.
KNOWN_CONFIG_KEYS: tuple[str, ...] = (
    "SPARKSAGE_API_KEY",
    "SPARKSAGE_BASE_URL",
    "SPARKSAGE_MODEL",
    "SPARKSAGE_STREAM",
    "SPARKSAGE_LANGUAGE",
    "SPARKSAGE_LOG_LEVEL",
    "SPARKSAGE_DATA_DIR",
    "SPARKSAGE_DOC_STORE",
    "SPARKSAGE_DOC_STORE_TABLE",
    "SPARKSAGE_KB_STORE",
    "SPARKSAGE_KB_STATE_STORE",
    "SPARKSAGE_FEEDBACK_STORE",
    "SPARKSAGE_AUTO_TAG_EXTRACTOR",
    "SPARKSAGE_AUTO_TAG_MIN_COHESION",
    "SPARKSAGE_TAGS_ZH",
    "SPARKSAGE_ENABLE_QA",
    "SPARKSAGE_EMBEDDING_API_KEY",
    "SPARKSAGE_EMBEDDING_BASE_URL",
    "SPARKSAGE_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)

#: A valid environment-variable name (mirrors :mod:`sparksage.config`).
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Sentinel value returned in place of a real secret.
_MASK = "****"


class ConfigError(ValueError):
    """Raised on an invalid write request (bad key name / value)."""


def mask_value(key: str, value: str) -> str:
    """Return the value to echo back for ``key`` (masked when sensitive).

    A non-empty value for a key ending in :data:`SENSITIVE_SUFFIX` becomes
    :data:`_MASK`. An empty secret stays empty so the UI can tell "unset" apart
    from "set but hidden".
    """
    if key.endswith(SENSITIVE_SUFFIX) and value:
        return _MASK
    return value


def read_config(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Report the effective value of every known configuration key.

    *Effective* means: a real environment variable wins over the ``.env`` file
    (the same precedence :func:`sparksage.config.load_dotenv` applies at
    startup), so the UI reflects what the running service actually uses.

    Secrets (keys ending in :data:`SENSITIVE_SUFFIX`) are passed through
    :func:`mask_value` before being returned, so the response is safe to send
    to a browser.

    A missing file is treated as "all keys unset" -- the reader never raises
    for a missing file, mirroring :func:`sparksage.config.load_dotenv`.
    """
    resolved = _resolve_path(path)
    file_values = _safe_parse(resolved)
    out: dict[str, str] = {}
    for key in KNOWN_CONFIG_KEYS:
        real = os.environ.get(key)
        value = real if real else file_values.get(key, "")
        out[key] = mask_value(key, value)
    return out


def write_config(
    updates: dict[str, str],
    path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Persist ``updates`` into the ``.env`` file, returning the applied keys.

    Only the supplied keys are touched -- comments, ordering and any unrelated
    keys already in the file are preserved. Keys not currently present are
    appended. A ``"****"`` value (the read-back sentinel for a secret) is
    treated as "no change" so a round-trip GET -> POST never clobbers a secret
    with the mask.

    Parameters
    ----------
    updates:
        ``{KEY: value}`` mapping to apply. Keys must be valid env-var names and
        restricted to the SparkSage/OpenAI surface (any ``SPARKSAGE_*``,
        ``OPENAI_*`` or ``*_API_KEY`` key). Values are coerced to ``str``.

    Raises
    ------
    ConfigError
        If a key is invalid or outside the allowed surface, or the file cannot
        be written.
    """
    resolved = _resolve_path(path)
    cleaned = _validate_updates(updates)

    if not cleaned:
        return []

    text = resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
    new_text = _apply_updates(text, cleaned)
    resolved.write_text(new_text, encoding="utf-8")

    # Keep the running process in sync (write-through), but never override a
    # real env var already set in the container/CI -- same 12-factor rule as
    # load_dotenv. This matches what a restart would apply from the file.
    for key, value in cleaned.items():
        if key not in os.environ:
            os.environ[key] = value
    return list(cleaned.keys())


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path) if path is not None else Path(DEFAULT_ENV_FILENAME)


def _safe_parse(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    from sparksage.config import parse_env_file

    try:
        return parse_env_file(path)
    except Exception:
        return {}


def _validate_updates(updates: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in updates.items():
        key = str(key).strip()
        if not _VALID_NAME_RE.match(key):
            raise ConfigError(f"invalid variable name: {key!r}")
        if not _is_allowed_key(key):
            raise ConfigError(
                f"refusing to write unknown key {key!r} "
                "(only SPARKSAGE_*, OPENAI_* and *_API_KEY keys are accepted)"
            )
        sval = "" if value is None else str(value)
        if key.endswith(SENSITIVE_SUFFIX) and sval == _MASK:
            continue
        cleaned[key] = sval
    return cleaned


def _is_allowed_key(key: str) -> bool:
    return (
        key.startswith("SPARKSAGE_")
        or key.startswith("OPENAI_")
        or key.endswith(SENSITIVE_SUFFIX)
    )


def _apply_updates(text: str, updates: dict[str, str]) -> str:
    """Rewrite ``text`` applying ``updates`` in place, preserving everything else.

    For each key already in the file the first ``KEY=...`` line is replaced (a
    quoted or unquoted value, optional ``export`` prefix, optional trailing
    comment). Keys absent from the file are appended. Comments, blank lines and
    unknown keys are left untouched.
    """
    seen: set[str] = set()
    out_lines: list[str] = []
    pattern = re.compile(
        r"^(?P<indent>\s*)(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=.*$"
    )
    for raw in text.splitlines():
        m = pattern.match(raw)
        if m and m.group("key") in updates:
            key = m.group("key")
            indent = m.group("indent")
            out_lines.append(f"{indent}{key}={_format_value(updates[key])}")
            seen.add(key)
        else:
            out_lines.append(raw)

    missing = [k for k in updates if k not in seen]
    if missing:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        elif not out_lines:
            out_lines.append("# SparkSage configuration (managed by the WEB UI).")
        for key in missing:
            out_lines.append(f"{key}={_format_value(updates[key])}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def _format_value(value: str) -> str:
    """Quote a value when it contains whitespace or a ``#`` (comment risk)."""
    if value == "":
        return ""
    if any(ch in value for ch in (" ", "\t", "#")) or value != value.strip():
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value
