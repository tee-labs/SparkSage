"""Logging configuration driven by the ``SPARKSAGE_LOG_LEVEL`` environment variable.

SparkSage uses the stdlib :mod:`logging` module throughout (every sub-package
gets a child logger under the ``sparksage`` namespace). By default the
``sparksage`` logger inherits its level from the root logger (``WARNING``), so
the library is quiet in production. Setting :data:`ENV_LOG_LEVEL` lets you turn
the verbosity up for debugging / analysis **without touching code**:

.. code-block:: console

    SPARKSAGE_LOG_LEVEL=DEBUG python -m sparksage.api.app
    SPARKSAGE_LOG_LEVEL=INFO uvicorn sparksage.api.app:create_app --factory

This module is **pure stdlib** (no third-party logging dependency), matches the
12-factor convention (real env vars win over ``.env``; :func:`load_dotenv` is
called before :func:`configure_logging` by :func:`sparksage.api.app.build_default_service`),
and deliberately does **not** configure logging on import -- libraries should
never mutate global logging state as a side effect of being imported. Call
:func:`configure_logging` explicitly at startup (the API entry point already
does this).

Design notes
------------

* Only the ``sparksage`` logger level is changed, so host / framework logging
  (uvicorn, gunicorn, the caller's own ``basicConfig``) is left untouched.
* A single :class:`~logging.StreamHandler` is installed on the ``sparksage``
  logger **only when nobody else has configured logging** (i.e. the root logger
  has no handlers). This makes ``SPARKSAGE_LOG_LEVEL=DEBUG`` "just work" in a
  plain script, while under uvicorn (which pre-configures root) records simply
  propagate up and are formatted by the host -- no duplicate output.
* :func:`configure_logging` is idempotent: calling it repeatedly never stacks
  handlers, so it is safe to re-run (e.g. on a service rebuild).
"""

from __future__ import annotations

import logging
import os
from typing import Final

__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "ENV_LOG_LEVEL",
    "LogLevelError",
    "configure_logging",
    "parse_level",
]

#: Environment variable read by :func:`configure_logging`.
ENV_LOG_LEVEL = "SPARKSAGE_LOG_LEVEL"

#: Level used when neither the env var nor an explicit arg is provided. Matches
#: the stdlib root-logger default so the library is silent at WARNING+ unless
#: the user opts in.
DEFAULT_LOG_LEVEL = "WARNING"

#: Logger whose level is controlled by :data:`ENV_LOG_LEVEL`. All SparkSage
#: sub-loggers (``sparksage.api``, ``sparksage.convert``, ...) are children of
#: this one, so a single level covers the whole library.
_LOGGER_NAME = "sparksage"

#: Format used by the fallback handler installed in "no host logging" mode.
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: The six canonical :mod:`logging` level names.
_VALID_LEVEL_NAMES: Final[frozenset[str]] = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)


class LogLevelError(ValueError):
    """Raised when a log level name or value cannot be resolved."""


def parse_level(level: str | int) -> int:
    """Resolve a level name (case-insensitive) or numeric value to a numeric level.

    Accepts the canonical names (``"DEBUG"``, ``"info"``, ...) as well as their
    numeric equivalents (``10``, ``20``, ...). A leading ``"="`` (as emitted by
    some tools, e.g. ``--log-level=DEBUG``) is tolerated.

    Raises
    ------
    LogLevelError
        If ``level`` is empty or not a recognised name / number.
    """
    if isinstance(level, int):
        if level < 0:
            raise LogLevelError(f"invalid log level: {level!r} (must be >= 0)")
        return level

    name = level.strip().lstrip("=").upper()
    if not name:
        raise LogLevelError("empty log level")

    if name.isdigit():
        numeric = int(name)
        if numeric < 0:
            raise LogLevelError(f"invalid log level: {level!r} (must be >= 0)")
        return numeric

    if name not in _VALID_LEVEL_NAMES:
        raise LogLevelError(
            f"unknown log level: {level!r} "
            f"(expected one of {sorted(_VALID_LEVEL_NAMES)} or an integer)"
        )
    return logging.getLevelName(name)


def configure_logging(level: str | int | None = None) -> int:
    """Configure the ``sparksage`` logger from :data:`ENV_LOG_LEVEL`.

    Resolution order (first wins):

    1. The explicit ``level`` argument (handy for tests / programmatic setup).
    2. The :data:`ENV_LOG_LEVEL` environment variable.
    3. :data:`DEFAULT_LOG_LEVEL` (``"WARNING"``).

    Sets the resolved level on the ``sparksage`` logger. When *no other logging
    has been configured* (the root logger has no handlers) a single
    :class:`~logging.StreamHandler` is attached so that INFO/DEBUG output is
    actually visible -- otherwise records would be silently dropped by Python's
    last-resort handler (which is capped at WARNING). Under a host that already
    configures logging (uvicorn, gunicorn, the caller's ``basicConfig``) no
    handler is added and records propagate up to the host's handlers, avoiding
    duplicate output.

    Idempotent: safe to call repeatedly; it never stacks handlers.

    Returns the resolved numeric level.
    """
    if level is None:
        raw = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    else:
        raw = level
    numeric = parse_level(raw)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric)

    root = logging.getLogger()
    if not root.hasHandlers() and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        # Handler level left at NOTSET so the logger level stays the single
        # knob: re-calling configure_logging() with a new level Just Works,
        # without having to re-sync each installed handler.
        logger.addHandler(handler)

    return numeric
