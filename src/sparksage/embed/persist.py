"""Zero-dependency persistence for :class:`~sparksage.embed.store.InMemoryVectorStore`.

Embeddings are expensive to compute, so they should survive across process
restarts: generate once, :meth:`~sparksage.embed.store.InMemoryVectorStore.add`
into a store, :func:`save_store` it to disk, and :func:`load_store` it back next
run. The Distill pipeline and any long-running RAG service consume the reloaded
store the same way they consume a freshly built one -- both are just
:class:`~sparksage.embed.store.VectorStore` instances.

The on-disk format is plain **JSON** (human-inspectable, zero-dependency, the
same "no third-party library" stance as :mod:`sparksage.config`). ``numpy`` /
``faiss`` binary formats are deliberately deferred to the future ``[distill]``
group, where they earn their weight on million-vector corpora; for the
thousands-of-blocks scale the embedding layer targets, JSON is simple and
correct.

Format (versioned, so the loader can evolve)::

    {
      "format": "sparksage-vector-store",
      "version": 1,
      "dimension": 128,
      "vectors": {"<block_id>": [0.1, -0.2, ...], ...}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparksage.embed.store import InMemoryVectorStore

#: Magic string identifying a serialized SparkSage vector store.
STORE_FORMAT = "sparksage-vector-store"

#: On-disk schema version. Bump when the layout changes; the loader refuses
#: unknown future versions rather than guessing.
STORE_VERSION = 1


def save_store(store: InMemoryVectorStore, path: str | Path) -> Path:
    """Serialize ``store`` to ``path`` as JSON; return the resolved path.

    Writing is atomic-ish: the payload is built fully in memory and written in
    one :func:`json.dump` call, so a partially-written file (from e.g. a
    mid-write crash) is unlikely. The parent directory is created if missing.

    Parameters
    ----------
    store:
        The :class:`InMemoryVectorStore` to serialize (its current contents are
        snapshotted; later mutations to the store are not reflected on disk).
    path:
        Destination file path. The ``.json`` extension is conventional but not
        enforced.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": STORE_FORMAT,
        "version": STORE_VERSION,
        "dimension": store.dimension,
        "vectors": store.vectors(),
    }
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return dest


def load_store(path: str | Path) -> InMemoryVectorStore:
    """Load an :class:`InMemoryVectorStore` previously written by :func:`save_store`.

    The format marker and version are validated, so a corrupted or foreign file
    fails fast with a clear error instead of silently producing a broken store.

    Parameters
    ----------
    path:
        File previously produced by :func:`save_store`.
    """
    src = Path(path)
    with src.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"invalid vector-store file {src!s}: expected a JSON object")
    fmt = payload.get("format")
    if fmt != STORE_FORMAT:
        raise ValueError(
            f"invalid vector-store file {src!s}: format marker is {fmt!r}, "
            f"expected {STORE_FORMAT!r}"
        )
    version = payload.get("version")
    if version != STORE_VERSION:
        raise ValueError(
            f"unsupported vector-store version {version!r} in {src!s}: "
            f"this build understands version {STORE_VERSION}"
        )
    dimension = payload.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError(f"invalid dimension {dimension!r} in vector-store file {src!s}")
    raw_vectors = payload.get("vectors")
    if not isinstance(raw_vectors, dict):
        raise ValueError(f"invalid vectors in {src!s}: expected a JSON object")

    store = InMemoryVectorStore(dimension=dimension)
    for block_id, vec in raw_vectors.items():
        if not isinstance(vec, list):
            raise ValueError(
                f"invalid vector for block {block_id!r} in {src!s}: expected a list"
            )
        store.add(str(block_id), [float(x) for x in vec])
    return store
