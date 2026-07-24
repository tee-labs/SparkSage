"""Tests for the ``.env`` loader (:mod:`sparksage.config`).

Pure-stdlib, fully offline. ``load_dotenv`` mutates :data:`os.environ`, so every
test that touches it uses ``monkeypatch`` to set/restore keys so the suite stays
hermetic regardless of the host environment or any real ``.env`` on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparksage.config import (
    DEFAULT_ENV_FILENAME,
    EnvParseError,
    load_dotenv,
    parse_env_file,
)


# ---------------------------------------------------------------------------- #
# parse_env_file
# ---------------------------------------------------------------------------- #
def test_parse_basic_key_value(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO=bar\n")
    assert parse_env_file(f) == {"FOO": "bar"}


def test_parse_strips_surrounding_whitespace(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO = bar \n")
    assert parse_env_file(f) == {"FOO": "bar"}


@pytest.mark.parametrize("quote", ['"', "'"])
def test_parse_quoted_values(tmp_path: Path, quote: str) -> None:
    f = tmp_path / ".env"
    f.write_text(f"FOO={quote}hello world{quote}\n")
    assert parse_env_file(f) == {"FOO": "hello world"}


def test_parse_quoted_value_keeps_internal_spaces_and_special(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('URL="https://host/path?x=1&y=2"\n')
    assert parse_env_file(f) == {"URL": "https://host/path?x=1&y=2"}


def test_parse_quoted_value_followed_by_inline_comment(tmp_path: Path) -> None:
    # Trailing content after a closed quote (e.g. a comment) must be ignored,
    # not mistaken for an unterminated quote.
    f = tmp_path / ".env"
    f.write_text('MODEL="gpt-4o-mini"  # the default\n')
    assert parse_env_file(f) == {"MODEL": "gpt-4o-mini"}


def test_parse_quoted_value_contains_hash(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('TOKEN="ab#cd"\n')
    assert parse_env_file(f) == {"TOKEN": "ab#cd"}


def test_parse_export_prefix_ignored(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("export FOO=bar\n")
    assert parse_env_file(f) == {"FOO": "bar"}


def test_parse_full_line_comment_skipped(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("# a comment\nFOO=bar\n   # indented comment\n")
    assert parse_env_file(f) == {"FOO": "bar"}


def test_parse_blank_lines_skipped(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("\nFOO=bar\n\n\nBAZ=qux\n")
    assert parse_env_file(f) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_trailing_inline_comment_only_when_preceded_by_space(
    tmp_path: Path,
) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO=bar # a comment\n")
    assert parse_env_file(f) == {"FOO": "bar"}


def test_parse_hash_inside_unquoted_token_preserved(tmp_path: Path) -> None:
    # '#' not preceded by whitespace is part of the value (e.g. URL anchor).
    f = tmp_path / ".env"
    f.write_text("URL=https://host/#anchor\n")
    assert parse_env_file(f) == {"URL": "https://host/#anchor"}


def test_parse_duplicate_keys_last_wins(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO=1\nFOO=2\n")
    assert parse_env_file(f) == {"FOO": "2"}


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_env_file(tmp_path / "nope.env")


def test_parse_missing_equals_raises(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("NOT_AN_ASSIGNMENT\n")
    with pytest.raises(EnvParseError, match="KEY=VALUE"):
        parse_env_file(f)


def test_parse_invalid_key_name_raises(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("1FOO=bar\n")
    with pytest.raises(EnvParseError, match="invalid variable name"):
        parse_env_file(f)


def test_parse_invalid_key_with_dash_raises(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO-BAR=bar\n")
    with pytest.raises(EnvParseError, match="invalid variable name"):
        parse_env_file(f)


def test_parse_unterminated_quote_raises(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('FOO="unterminated\n')
    with pytest.raises(EnvParseError, match="unterminated"):
        parse_env_file(f)


def test_parse_error_message_includes_line_number(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("FOO=bar\nBAD\n")
    with pytest.raises(EnvParseError, match=":2:"):
        parse_env_file(f)


def test_parse_empty_value(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("EMPTY=\n")
    assert parse_env_file(f) == {"EMPTY": ""}


def test_parse_does_not_touch_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPARKSAGE_TEST_PARSE", raising=False)
    f = tmp_path / ".env"
    f.write_text("SPARKSAGE_TEST_PARSE=loaded\n")
    parse_env_file(f)
    import os

    assert "SPARKSAGE_TEST_PARSE" not in os.environ


# ---------------------------------------------------------------------------- #
# load_dotenv
# ---------------------------------------------------------------------------- #
def test_load_dotenv_applies_missing_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPARKSAGE_TEST_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("SPARKSAGE_TEST_KEY=from-file\n")
    applied = load_dotenv(f)
    import os

    assert applied == {"SPARKSAGE_TEST_KEY": "from-file"}
    assert os.environ["SPARKSAGE_TEST_KEY"] == "from-file"


def test_load_dotenv_does_not_override_existing_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARKSAGE_TEST_KEY", "from-env")
    f = tmp_path / ".env"
    f.write_text("SPARKSAGE_TEST_KEY=from-file\n")
    applied = load_dotenv(f)
    import os

    assert applied == {}
    assert os.environ["SPARKSAGE_TEST_KEY"] == "from-env"


def test_load_dotenv_override_true_clobbers_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARKSAGE_TEST_KEY", "from-env")
    f = tmp_path / ".env"
    f.write_text("SPARKSAGE_TEST_KEY=from-file\n")
    applied = load_dotenv(f, override=True)
    import os

    assert applied == {"SPARKSAGE_TEST_KEY": "from-file"}
    assert os.environ["SPARKSAGE_TEST_KEY"] == "from-file"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "does-not-exist.env") == {}


def test_load_dotenv_default_path_is_cwd_env_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPARKSAGE_TEST_CWD", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / DEFAULT_ENV_FILENAME).write_text("SPARKSAGE_TEST_CWD=from-cwd\n")
    applied = load_dotenv()
    import os

    assert applied == {"SPARKSAGE_TEST_CWD": "from-cwd"}
    assert os.environ["SPARKSAGE_TEST_CWD"] == "from-cwd"


def test_load_dotenv_real_env_takes_priority_over_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12-factor priority: a real env var must win over the .env file."""
    monkeypatch.setenv("SPARKSAGE_API_KEY", "real-secret")
    f = tmp_path / ".env"
    f.write_text("SPARKSAGE_API_KEY=file-secret\n")
    load_dotenv(f)
    import os

    assert os.environ["SPARKSAGE_API_KEY"] == "real-secret"


def test_load_dotenv_propagates_parse_error(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("BAD LINE\n")
    with pytest.raises(EnvParseError):
        load_dotenv(f)


# ---------------------------------------------------------------------------- #
# Integration: build_default_service picks up .env
# ---------------------------------------------------------------------------- #
def test_build_default_service_reads_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_default_service() loads a .env in CWD before reading env vars."""
    pytest.importorskip("markitdown")
    pytest.importorskip("openai")
    from sparksage.api.app import ENV_API_KEY, build_default_service

    monkeypatch.chdir(tmp_path)
    for key in (ENV_API_KEY, "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / DEFAULT_ENV_FILENAME).write_text(
        f"{ENV_API_KEY}=from-dotenv\nSPARKSAGE_MODEL=test-model\n"
    )

    svc = build_default_service()
    # A generator should be wired because the .env supplied an API key.
    assert svc.has_generator


def test_build_default_service_env_var_beats_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real env var must take priority over a same-named .env entry."""
    pytest.importorskip("markitdown")
    pytest.importorskip("openai")
    from sparksage.api.app import ENV_API_KEY, build_default_service

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_API_KEY, "real-env-secret")
    (tmp_path / DEFAULT_ENV_FILENAME).write_text(f"{ENV_API_KEY}=file-secret\n")

    svc = build_default_service()
    assert svc.has_generator


# ---------------------------------------------------------------------------- #
# _env_bool (stream / boolean env-var parsing)
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "True", "yes", "on"])
def test_env_bool_truthy(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from sparksage.api.app import _env_bool

    monkeypatch.setenv("SPARKSAGE_TEST_BOOL", raw)
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=False) is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "off"])
def test_env_bool_falsy(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from sparksage.api.app import _env_bool

    monkeypatch.setenv("SPARKSAGE_TEST_BOOL", raw)
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=True) is False


def test_env_bool_unset_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from sparksage.api.app import _env_bool

    monkeypatch.delenv("SPARKSAGE_TEST_BOOL", raising=False)
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=True) is True
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=False) is False


def test_env_bool_empty_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from sparksage.api.app import _env_bool

    monkeypatch.setenv("SPARKSAGE_TEST_BOOL", "")
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=True) is True


def test_env_bool_unrecognized_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from sparksage.api.app import _env_bool

    monkeypatch.setenv("SPARKSAGE_TEST_BOOL", "maybe")
    assert _env_bool("SPARKSAGE_TEST_BOOL", default=True) is True
