"""Demo: knowledge-document tag management (offline, no API key).

This demonstrates the document-management subsystem end-to-end using a
deterministic fake converter, so it runs with no optional dependencies and no
network:

    uploaded bytes -> convert -> clean -> parse -> auto-tag -> store -> CRUD

Key idea: when a document ships without tags, the system auto-generates them
from content via a **keyword-extraction algorithm** (pure stdlib -- no LLM).
Run with:  PYTHONPATH=src python3 examples/manage_documents.py
"""

from __future__ import annotations

from sparksage import (
    DEFAULT_AUTO_TAG_COUNT,
    DocumentService,
    FakeConverterBackend,
    MarkdownConverter,
)
from sparksage.schema.enums import TagSource

ONBOARDING = """\
# Onboarding Guide

This guide explains the onboarding workflow for new engineers joining the
platform team. It covers account setup, repository access and CI pipeline.

The onboarding workflow also documents the machine learning model training
environment and the deployment pipeline used by the platform team.
"""


def main() -> None:
    service = DocumentService(
        converter=MarkdownConverter(backend=FakeConverterBackend(markdown=ONBOARDING)),
    )

    # 1) Upload WITHOUT tags -> auto-generated from content (keyword extraction).
    auto = service.upload(b"raw bytes", "onboarding.md")
    print("== auto-tagged ==")
    print("title   :", auto.title)
    print("summary :", auto.summary)
    print("tags    :", auto.tags)
    print("source  :", auto.tag_source.value, "(auto-extracted)\n")

    # 2) Upload WITH tags -> user tags win, no extraction runs.
    manual = service.upload(
        b"raw bytes", "design.md", tags=["architecture", "review"]
    )
    print("== user-tagged ==")
    print("tags    :", manual.tags)
    print("source  :", manual.tag_source.value, "(user-supplied)\n")

    # 3) Merge auto-tags onto a user-tagged document -> MIXED provenance.
    merged = service.extract_tags(manual.id, top_n=DEFAULT_AUTO_TAG_COUNT)
    print("== merged auto + user ==")
    print("tags    :", merged.tags)
    print("source  :", merged.tag_source.value, "\n")

    # 4) List, filter by tag, and inspect the tag index.
    print("== knowledge base ==")
    print("count   :", service.count())
    print("by tag  :", service.list(tag="architecture")[0].title)
    print("index   :", service.tags())

    # 5) Update / delete.
    updated = service.update(auto.id, title="Engineering Onboarding")
    print("\n== after edit ==")
    print("title   :", updated.title, "| version", updated.version)
    service.delete(auto.id)
    print("count   :", service.count(), "(after delete)")


if __name__ == "__main__":
    assert TagSource  # ensure import side-effect / public name
    main()
