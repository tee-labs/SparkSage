"""Demo: manage documents with the in-memory document store.

Runs fully offline (no API key, no optional dependencies). Exercises the full
DocumentStore CRUD + tag-filtering contract on an InMemoryDocumentStore.

For durable storage, swap in a SqliteDocumentStore (single-file, no server):

    from sparksage import SqliteDocumentStore
    store = SqliteDocumentStore("./docs.db")

Run with:  PYTHONPATH=src python3 examples/manage_documents.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sparksage import InMemoryDocumentStore, SqliteDocumentStore, new_record


def exercise(store_label: str, store) -> None:
    print(f"\n=== {store_label} ===")

    record = store.save(new_record(
        title="Annual Report",
        body_markdown="# Annual Report\nRevenue grew 12% year over year.",
        tags=["revenue", "annual"],
        source="file://docs/annual_report.md",
    ))
    store.save(new_record(
        title="Expansion Plan",
        body_markdown="# Expansion\nThe company plans to expand to APAC.",
        tags=["strategy", "annual"],
        source="file://docs/expansion.md",
    ))

    print(f"count:      {len(store)}")
    print(f"contains:   {record.doc_id in store}")
    print(f"tags:       {store.list_tags()}")

    print("by tag=revenue:")
    for r in store.list(tag="revenue"):
        print(f"  - {r.doc_id[:8]}  {r.title}  tags={r.tags}")

    print(f"summary:    {record.summary!r}")
    print(f"hash:       {record.content_hash[:16]}...")

    deleted = store.delete(record.doc_id)
    print(f"deleted:    {deleted}  (count now {len(store)})")


def main() -> None:
    exercise("InMemoryDocumentStore", InMemoryDocumentStore())

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "docs.db"
        exercise(f"SqliteDocumentStore({db_path.name})", SqliteDocumentStore(str(db_path)))


if __name__ == "__main__":
    main()
