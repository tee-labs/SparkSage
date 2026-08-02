"""Tests for :mod:`sparksage.logging_config`.

Pure-stdlib, fully offline. :func:`configure_logging` mutates the global
``sparksage`` logger (and may attach a handler), so every test snapshots and
restores the logger's level / handlers / propagate and scrubs :data:`ENV_LOG_LEVEL`
via ``monkeypatch`` to keep the suite hermetic.
"""

from __future__ import annotations

import logging

import pytest

from sparksage.logging_config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    ENV_LOG_LEVEL,
    UVICORN_ACCESS_LOG_FORMAT,
    LogLevelError,
    build_uvicorn_log_config,
    configure_logging,
    parse_level,
)

_LOGGER = logging.getLogger("sparksage")


@pytest.fixture(autouse=True)
def _restore_sparksage_logger():
    """Snapshot/restore level, handlers and propagate on the sparksage logger."""
    saved_level = _LOGGER.level
    saved_handlers = list(_LOGGER.handlers)
    saved_propagate = _LOGGER.propagate
    yield
    _LOGGER.handlers = saved_handlers
    _LOGGER.setLevel(saved_level)
    _LOGGER.propagate = saved_propagate


@pytest.fixture(autouse=True)
def _scrub_log_level_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)


# ---------------------------------------------------------------------------- #
# parse_level
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("CRITICAL", logging.CRITICAL),
        ("ERROR", logging.ERROR),
        ("WARNING", logging.WARNING),
        ("INFO", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("NOTSET", logging.NOTSET),
    ],
)
def test_parse_level_names(name: str, expected: int) -> None:
    assert parse_level(name) == expected


@pytest.mark.parametrize("name", ["debug", "Info", "WARNING", "critical"])
def test_parse_level_is_case_insensitive(name: str) -> None:
    assert parse_level(name) == logging.getLevelName(name.upper())


@pytest.mark.parametrize("value", ["0", "10", "20", "30", "40", "50"])
def test_parse_level_accepts_numeric_strings(value: str) -> None:
    assert parse_level(value) == int(value)


@pytest.mark.parametrize("value", [0, 10, 20, 50])
def test_parse_level_accepts_ints(value: int) -> None:
    assert parse_level(value) == value


def test_parse_level_tolerates_leading_equals() -> None:
    assert parse_level("=DEBUG") == logging.DEBUG


@pytest.mark.parametrize("bad", ["", "   ", "VERBOSE", "garbage", "-5"])
def test_parse_level_rejects_unknown(bad: str) -> None:
    with pytest.raises(LogLevelError):
        parse_level(bad)


def test_parse_level_rejects_negative_int() -> None:
    with pytest.raises(LogLevelError):
        parse_level(-1)


# ---------------------------------------------------------------------------- #
# configure_logging: level resolution
# ---------------------------------------------------------------------------- #
def test_configure_logging_default_is_warning() -> None:
    assert configure_logging() == logging.WARNING
    assert _LOGGER.level == logging.WARNING


def test_configure_logging_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")
    assert configure_logging() == logging.DEBUG
    assert _LOGGER.level == logging.DEBUG


def test_configure_logging_env_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_LEVEL, "info")
    assert configure_logging() == logging.INFO


def test_configure_logging_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")
    assert configure_logging("ERROR") == logging.ERROR
    assert _LOGGER.level == logging.ERROR


def test_configure_logging_accepts_int_level() -> None:
    assert configure_logging(logging.DEBUG) == logging.DEBUG
    assert _LOGGER.level == logging.DEBUG


def test_configure_logging_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LOG_LEVEL, "VERBOSE")
    with pytest.raises(LogLevelError):
        configure_logging()


# ---------------------------------------------------------------------------- #
# configure_logging: handler installation
# ---------------------------------------------------------------------------- #
def test_configure_logging_installs_handler_when_root_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In plain-library mode (root has no handlers) a StreamHandler is attached."""
    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    root.handlers.clear()
    try:
        # sparksage logger must start clean too.
        _LOGGER.handlers.clear()
        configure_logging("DEBUG")
        assert len(_LOGGER.handlers) == 1
        handler = _LOGGER.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        # Handler level left at NOTSET so the logger level is the single knob.
        assert handler.level == logging.NOTSET
        assert isinstance(handler.formatter, logging.Formatter)
        assert handler.formatter._fmt == DEFAULT_LOG_FORMAT
    finally:
        root.handlers = saved_root_handlers


def test_configure_logging_does_not_install_handler_when_root_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a host that already configured root (uvicorn/basicConfig) we propagate."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    saved_root_handlers = list(root.handlers)
    root.handlers = [sentinel]
    _LOGGER.handlers.clear()
    try:
        configure_logging("DEBUG")
        assert _LOGGER.handlers == []
        # Propagation left on so records reach the host's root handlers.
        assert _LOGGER.propagate is True
    finally:
        root.handlers = saved_root_handlers


def test_configure_logging_is_idempotent_no_stacked_handlers() -> None:
    _LOGGER.handlers.clear()
    configure_logging("INFO")
    configure_logging("DEBUG")
    configure_logging("WARNING")
    assert len(_LOGGER.handlers) <= 1


def test_configure_logging_updates_logger_level_on_recall() -> None:
    """Re-calling with a different level moves the logger level (no stacking)."""
    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    root.handlers.clear()
    _LOGGER.handlers.clear()
    try:
        configure_logging("ERROR")
        assert _LOGGER.level == logging.ERROR
        configure_logging("DEBUG")
        assert _LOGGER.level == logging.DEBUG
        assert len(_LOGGER.handlers) == 1
    finally:
        root.handlers = saved_root_handlers


# ---------------------------------------------------------------------------- #
# End-to-end: a DEBUG record is actually emitted at DEBUG level
# ---------------------------------------------------------------------------- #
def test_debug_record_reaches_handler(caplog: pytest.LogCaptureFixture) -> None:
    """Sanity check: lowering the level surfaces previously-suppressed records."""
    # caplog attaches a handler to root -> configure_logging won't add its own,
    # and the records propagate up to caplog's handler.
    configure_logging("DEBUG")
    logging.getLogger("sparksage.convert").debug("hello-distill")
    assert any("hello-distill" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------- #
# Public surface
# ---------------------------------------------------------------------------- #
def test_constants_exposed() -> None:
    assert DEFAULT_LOG_LEVEL == "WARNING"
    assert ENV_LOG_LEVEL == "SPARKSAGE_LOG_LEVEL"


def test_exported_from_top_level_package() -> None:
    import sparksage

    assert sparksage.ENV_LOG_LEVEL == "SPARKSAGE_LOG_LEVEL"
    assert sparksage.DEFAULT_LOG_LEVEL == "WARNING"
    assert sparksage.configure_logging is configure_logging
    assert sparksage.parse_level is parse_level
    assert issubclass(sparksage.LogLevelError, ValueError)


# ---------------------------------------------------------------------------- #
# build_uvicorn_log_config: unify uvicorn access / app log formats
# ---------------------------------------------------------------------------- #
def test_build_uvicorn_log_config_default_formatter_matches_app() -> None:
    cfg = build_uvicorn_log_config()
    assert cfg["disable_existing_loggers"] is False
    assert cfg["formatters"]["default"]["format"] == DEFAULT_LOG_FORMAT


def test_build_uvicorn_log_config_access_formatter_uses_app_shape() -> None:
    cfg = build_uvicorn_log_config()
    access = cfg["formatters"]["access"]
    # Resolved lazily inside uvicorn (no uvicorn import here) -> pure stdlib.
    assert access["()"] == "uvicorn.logging.AccessFormatter"
    assert access["fmt"] == UVICORN_ACCESS_LOG_FORMAT
    # Same timestamp / level / name prefix as the application format; the
    # access fields take the place of ``%(message)s`` in the body.
    assert UVICORN_ACCESS_LOG_FORMAT.startswith(DEFAULT_LOG_FORMAT.replace("%(message)s", ""))


def test_build_uvicorn_log_config_keeps_uvicorn_logger_topology() -> None:
    cfg = build_uvicorn_log_config()
    loggers = cfg["loggers"]
    # Non-propagating, each with its own handler -> no duplicate output.
    assert loggers["uvicorn"]["propagate"] is False
    assert loggers["uvicorn.access"]["propagate"] is False
    assert loggers["uvicorn"]["handlers"] == ["default"]
    assert loggers["uvicorn.access"]["handlers"] == ["access"]
    # Default level matches uvicorn's own (request logs always emitted).
    assert loggers["uvicorn"]["level"] == logging.INFO
    assert loggers["uvicorn.access"]["level"] == logging.INFO


def test_build_uvicorn_log_config_level_override() -> None:
    cfg = build_uvicorn_log_config("DEBUG")
    assert cfg["loggers"]["uvicorn"]["level"] == logging.DEBUG
    assert cfg["loggers"]["uvicorn.access"]["level"] == logging.DEBUG
