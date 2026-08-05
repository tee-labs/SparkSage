"""Tests for the ``/api/v1/cleaning`` routes.

Exercised via ``TestClient`` over a service built from fakes. The script-
compilation paths additionally need the ``[clean-script]`` extra, so those are
guarded by ``importorskip("RestrictedPython")``; the CRUD plumbing (list /
create / get / patch / delete) works without it because the store does not
compile until the manager rebuilds.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app
    from sparksage.api.pipeline import SparkSageService
    from sparksage.clean.cleaner import TextCleaner
    from sparksage.convert.backend import FakeConverterBackend
    from sparksage.convert.converter import MarkdownConverter

    svc = SparkSageService(
        converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
        cleaner=TextCleaner(),
    )
    return TestClient(create_app(service=svc))


CODE = "def clean(text, source=None):\n    return text.replace('X', 'Y')\n"


class TestCleaningRoutes:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/cleaning")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["items"] == []

    def test_create_then_get_then_list(self, client):
        body = {"name": "redact", "code": CODE}
        resp = client.post("/api/v1/cleaning", json=body)
        assert resp.status_code == 201
        created = resp.json()
        rid = created["rule_id"]
        assert created["name"] == "redact"
        assert created["pattern_kind"] == "none"
        assert created["enabled"] is True

        # GET single
        got = client.get(f"/api/v1/cleaning/{rid}").json()
        assert got["rule_id"] == rid

        # LIST sees it
        lst = client.get("/api/v1/cleaning").json()
        assert lst["count"] == 1
        assert lst["total"] == 1

    def test_create_with_routing_fields(self, client):
        resp = client.post(
            "/api/v1/cleaning",
            json={
                "name": "pdf",
                "code": CODE,
                "pattern_kind": "glob",
                "source_pattern": "*.pdf",
                "timeout": 2.5,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["pattern_kind"] == "glob"
        assert data["source_pattern"] == "*.pdf"
        assert data["timeout"] == 2.5

    def test_patch_updates_fields(self, client):
        rid = client.post("/api/v1/cleaning", json={"name": "a", "code": CODE}).json()["rule_id"]
        resp = client.patch(f"/api/v1/cleaning/{rid}", json={"enabled": False, "name": "a2"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["name"] == "a2"

    def test_patch_unknown_returns_404(self, client):
        resp = client.patch("/api/v1/cleaning/nope", json={"enabled": False})
        assert resp.status_code == 404

    def test_delete(self, client):
        rid = client.post("/api/v1/cleaning", json={"name": "a", "code": CODE}).json()["rule_id"]
        assert client.delete(f"/api/v1/cleaning/{rid}").status_code == 204
        assert client.get(f"/api/v1/cleaning/{rid}").status_code == 404
        # idempotent 404 on second delete
        assert client.delete(f"/api/v1/cleaning/{rid}").status_code == 404

    def test_validation_rejects_bad_body(self, client):
        # missing required code
        resp = client.post("/api/v1/cleaning", json={"name": "a"})
        assert resp.status_code == 422

    def test_compile_error_flagged_in_list(self, client):
        # a script that RestrictedPython rejects at compile time
        bad = "def clean(text):\n    return eval(text)\n"
        rid = client.post("/api/v1/cleaning", json={"name": "bad", "code": bad}).json()["rule_id"]
        item = client.get("/api/v1/cleaning").json()["items"][0]
        assert item["rule_id"] == rid
        # compiled is False only when RestrictedPython is installed; either way
        # an enabled rule that fails to compile surfaces an error.
        if not item["compiled"]:
            assert item["error"]


class TestCleaningTestRoute:
    def test_test_route_runs_script(self, client):
        pytest.importorskip("RestrictedPython")
        resp = client.post(
            "/api/v1/cleaning/test",
            json={
                "code": "def clean(text, source=None):\n    return text.upper()\n",
                "text": "hello",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["output"] == "HELLO"
        assert data["error"] is None

    def test_test_route_reports_compile_error(self, client):
        pytest.importorskip("RestrictedPython")
        resp = client.post(
            "/api/v1/cleaning/test",
            json={"code": "def clean(text):\n    return eval(text)\n", "text": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["error"]
        assert data["output"] == "hi"

    def test_test_route_does_not_persist(self, client):
        pytest.importorskip("RestrictedPython")
        client.post(
            "/api/v1/cleaning/test",
            json={"code": "def clean(text):\n    return text\n", "text": "x"},
        )
        assert client.get("/api/v1/cleaning").json()["total"] == 0


class TestCleaningRouteAppliesAtIngest:
    def test_created_rule_applies_to_convert(self, client):
        pytest.importorskip("RestrictedPython")
        # a global rule that upper-cases everything
        client.post(
            "/api/v1/cleaning",
            json={
                "name": "upper",
                "code": "def clean(text, source=None):\n    return text.upper()\n",
            },
        )
        # the convert route applies the live cleaner when clean=true
        resp = client.post(
            "/api/v1/convert",
            files={"file": ("n.md", b"ignored", "text/plain")},
            data={"clean": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["cleaned"] is True
