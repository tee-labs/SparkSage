"""Environment configuration: load ``.env`` files into :data:`os.environ`.

SparkSage reads its settings from environment variables (see
:func:`sparksage.api.app.build_default_service`). This module lets you provide
those defaults via a ``.env`` file -- the de-facto standard for local/service
configuration -- **without adding any third-party dependency**. The parser is
small, pure stdlib, and intentionally restricted to the well-defined subset of
``.env`` syntax (it deliberately does *not* implement shell expansion).

Priority (highest first), matching the 12-factor convention:

1. Real environment variables already set in the process (container / CI /
   system). These are never overwritten unless ``override=True``.
2. Values read from the ``.env`` file.

Supported ``.env`` syntax:

* ``KEY=VALUE``
* ``KEY="double quoted"`` / ``KEY='single quoted'``  (quotes stripped)
* ``export KEY=VALUE``                                (``export`` prefix ignored)
* ``# full-line comment`` and trailing ``KEY=value # comment``
  (a ``#`` is only treated as a comment when preceded by whitespace, so URLs
  like ``https://host/#anchor`` survive intact)
* blank lines

Deliberately unsupported (would require a shell parser and is a common source
of quoting bugs / injection risk): command substitution ``$(...)``, backticks,
variable interpolation ``$KEY`` / ``${KEY}``, multi-line values, and escape
sequences inside double quotes.

Keep secrets out of the repo: ``.env`` is git-ignored; commit a
``.env.example`` template instead (see the repo root).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "DEFAULT_ENV_FILENAME",
    "EnvParseError",
    "load_dotenv",
    "parse_env_file",
]

#: Default filename searched for in the current working directory.
DEFAULT_ENV_FILENAME = ".env"

#: A valid environment-variable name: ASCII letter/underscore start, then
#: letters, digits or underscores. Matches the POSIX ``name`` definition used
#: by shells and by :func:`os.environ`.
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvParseError(ValueError):
    """Raised when a ``.env`` line cannot be parsed."""


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse a ``.env`` file into a ``dict`` without touching :data:`os.environ`.

    Parameters
    ----------
    path:
        Path to the ``.env`` file. Must exist.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    EnvParseError
        If a line is malformed (no ``=``, invalid key name, unterminated quote).

    Returns a mapping of variable name -> parsed value, in file order.
    Duplicate keys keep the *last* definition (same semantics as shells).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise EnvParseError(f"{p}:{lineno}: expected 'KEY=VALUE', got {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not _VALID_NAME_RE.match(key):
            raise EnvParseError(
                f"{p}:{lineno}: invalid variable name {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_]*)"
            )
        try:
            result[key] = _parse_value(value)
        except EnvParseError as exc:
            raise EnvParseError(f"{p}:{lineno}: {exc}") from None
    return result


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load variables from a ``.env`` file into :data:`os.environ`.

    By default this only fills in variables that are **not already set** in the
    process environment, so real environment variables (container / CI / system)
    always win over the file -- this is the safe, expected behaviour. Pass
    ``override=True`` to let the file clobber existing values.

    A missing file is a no-op (returns ``{}``), so this is safe to call
    unconditionally at startup.

    Parameters
    ----------
    path:
        Path to the ``.env`` file. When ``None``, ``DEFAULT_ENV_FILENAME`` is
        looked up in the current working directory. No upward directory walk is
        performed -- pick up an explicit path if you keep your ``.env``
        elsewhere.
    override:
        If ``True``, values from the file overwrite existing environment
        variables. Default ``False``.

    Returns the mapping of variables that were actually applied to the
    environment (i.e. excluding those skipped because a real env var already had
    them and ``override`` is ``False``).
    """
    resolved = Path(path) if path is not None else Path(DEFAULT_ENV_FILENAME)
    if not resolved.is_file():
        return {}
    parsed = parse_env_file(resolved)
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _parse_value(raw: str) -> str:
    """Parse the RHS of ``KEY=VALUE`` into the final string value.

    * Quoted (single or double) -> the literal content between the first pair of
      matching quotes; anything after the closing quote (e.g. a trailing
      ``# comment``) is ignored.
    * Unquoted -> a trailing `` # comment`` is stripped (only when ``#`` is
      preceded by whitespace, so ``https://host/#anchor`` is preserved), then
      surrounding whitespace is trimmed.
    """
    value = raw.strip()
    if value and value[0] in "\"'":
        quote = value[0]
        close = value.find(quote, 1)
        if close == -1:
            raise EnvParseError(f"unterminated {quote!r}-quoted value")
        return value[1:close]
    return _strip_inline_comment(value)


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ``# comment`` from an unquoted value.

    Only a ``#`` preceded by whitespace is treated as a comment start, matching
    shell behaviour and keeping ``#`` inside unquoted tokens (URLs, hashes)
    intact.
    """
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1] in " \t":
            return value[:i].rstrip()
    return value
