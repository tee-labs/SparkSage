"""Tests for the ``/api/v1/config`` routes and the :mod:`config_manager`.

The config manager is pure stdlib so it is tested directly. The HTTP routes
are exercised via ``TestClient`` (guarded by ``importorskip``) over a service
built from fakes, with the ``.env`` path redirected into a temp directory.
"""

from __future__ import annotations

import pytest

from sparksage.api.config_manager import (
    ConfigError,
    mask_value,
    read_config,
    write_config,
)


# ---------------------------------------------------------------------------- #
# config_manager unit tests
# ---------------------------------------------------------------------------- #
class TestConfigManager:
    def test_read_returns_all_known_keys_empty(self, tmp_path, monkeypatch):
        for k in [
            "SPARKSAGE_API_KEY",
            "SPARKSAGE_MODEL",
            "SPARKSAGE_DOC_STORE",
        ]:
            monkeypatch.delenv(k, raising=False)
        out = read_config(path=tmp_path / ".env")
        assert "SPARKSAGE_API_KEY" in out
        assert out["SPARKSAGE_API_KEY"] == ""
        assert out["SPARKSAGE_MODEL"] == ""
        # every known key present
        from sparksage.api.config_manager import KNOWN_CONFIG_KEYS

        for key in KNOWN_CONFIG_KEYS:
            assert key in out

    def test_read_masks_sensitive_nonempty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPARKSAGE_API_KEY", "sk-secret")
        monkeypatch.delenv("SPARKSAGE_MODEL", raising=False)
        out = read_config(path=tmp_path / ".env")
        assert out["SPARKSAGE_API_KEY"] == "****"
        assert out["SPARKSAGE_MODEL"] == ""

    def test_read_real_env_wins_over_file(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("SPARKSAGE_MODEL=gpt-4o-mini\n", encoding="utf-8")
        monkeypatch.setenv("SPARKSAGE_MODEL", "gpt-4o")
        out = read_config(path=f)
        assert out["SPARKSAGE_MODEL"] == "gpt-4o"

    def test_read_file_value_when_env_unset(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("SPARKSAGE_MODEL=gpt-4o-mini\n", encoding="utf-8")
        monkeypatch.delenv("SPARKSAGE_MODEL", raising=False)
        out = read_config(path=f)
        assert out["SPARKSAGE_MODEL"] == "gpt-4o-mini"

    def test_mask_value_rules(self):
        assert mask_value("SPARKSAGE_API_KEY", "secret") == "****"
        assert mask_value("SPARKSAGE_API_KEY", "") == ""
        assert mask_value("SPARKSAGE_MODEL", "gpt-4o") == "gpt-4o"

    def test_write_creates_file_and_applies(self, tmp_path, monkeypatch):
        for k in ["SPARKSAGE_MODEL", "SPARKSAGE_LOG_LEVEL"]:
            monkeypatch.delenv(k, raising=False)
        f = tmp_path / ".env"
        applied = write_config(
            {"SPARKSAGE_MODEL": "gpt-4o", "SPARKSAGE_LOG_LEVEL": "DEBUG"}, path=f
        )
        assert applied == ["SPARKSAGE_MODEL", "SPARKSAGE_LOG_LEVEL"]
        text = f.read_text(encoding="utf-8")
        assert "SPARKSAGE_MODEL=gpt-4o" in text
        assert "SPARKSAGE_LOG_LEVEL=DEBUG" in text

    def test_write_preserves_comments_and_unknown_keys(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPARKSAGE_MODEL", raising=False)
        f = tmp_path / ".env"
        f.write_text(
            "# header comment\nOTHER_KEY=keepme\nSPARKSAGE_MODEL=gpt-4o-mini\n",
            encoding="utf-8",
        )
        write_config({"SPARKSAGE_MODEL": "gpt-4o"}, path=f)
        text = f.read_text(encoding="utf-8")
        assert "# header comment" in text
        assert "OTHER_KEY=keepme" in text
        assert "SPARKSAGE_MODEL=gpt-4o" in text
        assert "gpt-4o-mini" not in text

    def test_write_quotes_values_with_spaces(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPARKSAGE_BASE_URL", raising=False)
        f = tmp_path / ".env"
        write_config({"SPARKSAGE_BASE_URL": "https://host/v1"}, path=f)
        # no spaces -> unquoted
        assert "SPARKSAGE_BASE_URL=https://host/v1" in f.read_text(encoding="utf-8")
        write_config({"SPARKSAGE_LANGUAGE": "en US"}, path=f)
        assert 'SPARKSAGE_LANGUAGE="en US"' in f.read_text(encoding="utf-8")

    def test_write_secret_mask_is_noop(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("SPARKSAGE_API_KEY=sk-real\n", encoding="utf-8")
        applied = write_config({"SPARKSAGE_API_KEY": "****"}, path=f)
        assert applied == []
        assert "sk-real" in f.read_text(encoding="utf-8")

    def test_write_secret_real_value_overwrites(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("SPARKSAGE_API_KEY=sk-old\n", encoding="utf-8")
        applied = write_config({"SPARKSAGE_API_KEY": "sk-new"}, path=f)
        assert applied == ["SPARKSAGE_API_KEY"]
        assert "sk-new" in f.read_text(encoding="utf-8")
        assert "sk-old" not in f.read_text(encoding="utf-8")

    def test_write_rejects_unknown_key(self, tmp_path):
        with pytest.raises(ConfigError):
            write_config({"RANDOM_VAR": "x"}, path=tmp_path / ".env")

    def test_write_rejects_bad_name(self, tmp_path):
        with pytest.raises(ConfigError):
            write_config({"1BAD": "x"}, path=tmp_path / ".env")

    def test_write_roundtrips_through_read(self, tmp_path, monkeypatch):
        for k in ["SPARKSAGE_MODEL", "SPARKSAGE_LOG_LEVEL", "SPARKSAGE_API_KEY"]:
            monkeypatch.delenv(k, raising=False)
        f = tmp_path / ".env"
        write_config(
            {"SPARKSAGE_MODEL": "gpt-4o", "SPARKSAGE_API_KEY": "sk-abc"},
            path=f,
        )
        out = read_config(path=f)
        assert out["SPARKSAGE_MODEL"] == "gpt-4o"
        assert out["SPARKSAGE_API_KEY"] == "****"


# ---------------------------------------------------------------------------- #
# HTTP route integration tests
# ---------------------------------------------------------------------------- #
@pytest.fixture
def http_app(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app
    from sparksage.api.pipeline import SparkSageService
    from sparksage.clean.cleaner import TextCleaner
    from sparksage.convert.backend import FakeConverterBackend
    from sparksage.convert.converter import MarkdownConverter

    # isolate the .env from real env + CWD: the routes default to ".env" in CWD.
    for k in [
        "SPARKSAGE_API_KEY",
        "SPARKSAGE_MODEL",
        "SPARKSAGE_LOG_LEVEL",
        "SPARKSAGE_BASE_URL",
        "SPARKSAGE_LANGUAGE",
    ]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)

    svc = SparkSageService(
        converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
        cleaner=TextCleaner(),
    )
    app = create_app(service=svc)
    client = TestClient(app)
    return client, tmp_path, monkeypatch


class TestConfigRoutes:
    def test_get_config_returns_known_keys(self, http_app):
        client, tmp_path, _ = http_app
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "variables" in data
        assert "SPARKSAGE_MODEL" in data["variables"]
        assert data["variables"]["SPARKSAGE_MODEL"] == ""

    def test_post_config_writes_and_reports_restart(self, http_app):
        client, tmp_path, _ = http_app
        resp = client.post(
            "/api/v1/config",
            json={"SPARKSAGE_MODEL": "gpt-4o", "SPARKSAGE_LOG_LEVEL": "DEBUG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["restart_required"] is True
        assert set(data["applied"]) == {"SPARKSAGE_MODEL", "SPARKSAGE_LOG_LEVEL"}
        env_path = tmp_path / ".env"
        assert env_path.is_file()
        text = env_path.read_text(encoding="utf-8")
        assert "SPARKSAGE_MODEL=gpt-4o" in text

    def test_post_config_rejects_unknown_key(self, http_app):
        client, _, _ = http_app
        resp = client.post("/api/v1/config", json={"RANDOM_VAR": "x"})
        assert resp.status_code == 422

    def test_get_config_masks_secrets(self, http_app):
        client, tmp_path, _ = http_app
        (tmp_path / ".env").write_text(
            "SPARKSAGE_API_KEY=sk-secret\n", encoding="utf-8"
        )
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        assert resp.json()["variables"]["SPARKSAGE_API_KEY"] == "****"

    def test_get_config_real_env_wins(self, http_app, monkeypatch):
        client, _, _ = http_app
        monkeypatch.setenv("SPARKSAGE_MODEL", "gpt-4o")
        resp = client.get("/api/v1/config")
        assert resp.json()["variables"]["SPARKSAGE_MODEL"] == "gpt-4o"


# ---------------------------------------------------------------------------- #
# static frontend serving (SPA catch-all)
# ---------------------------------------------------------------------------- #
class TestStaticFrontend:
    def _app_with_dist(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from sparksage.api.app import create_app
        from sparksage.api.pipeline import SparkSageService
        from sparksage.clean.cleaner import TextCleaner
        from sparksage.convert.backend import FakeConverterBackend
        from sparksage.convert.converter import MarkdownConverter

        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
        assets = dist / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("console.log(1)", encoding="utf-8")

        monkeypatch.setenv("SPARKSAGE_WEB_DIST", str(dist))
        svc = SparkSageService(
            converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
            cleaner=TextCleaner(),
        )
        app = create_app(service=svc)
        return TestClient(app)

    def test_spa_root_served(self, tmp_path, monkeypatch):
        client = self._app_with_dist(tmp_path, monkeypatch)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SPA" in resp.text

    def test_spa_unknown_path_falls_back_to_index(self, tmp_path, monkeypatch):
        client = self._app_with_dist(tmp_path, monkeypatch)
        resp = client.get("/some/spa/route")
        assert resp.status_code == 200
        assert "SPA" in resp.text

    def test_spa_serves_real_asset(self, tmp_path, monkeypatch):
        client = self._app_with_dist(tmp_path, monkeypatch)
        resp = client.get("/assets/app.js")
        assert resp.status_code == 200
        assert "console.log" in resp.text

    def test_unknown_api_path_returns_404_not_405(self, tmp_path, monkeypatch):
        client = self._app_with_dist(tmp_path, monkeypatch)
        # GET + POST both 404 on an unknown api path (not 405 from catch-all)
        assert client.get("/api/v1/knowledge_base").status_code == 404
        assert client.post("/api/v1/query", json={"query": "x"}).status_code == 404

    def test_known_api_route_still_works(self, tmp_path, monkeypatch):
        client = self._app_with_dist(tmp_path, monkeypatch)
        assert client.get("/api/v1/health").status_code == 200
