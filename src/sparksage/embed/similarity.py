"""All-pairs near-duplicate detection over embedding vectors (pure stdlib).

The retrieval counterpart to :mod:`sparksage.embed.store`. The store answers
"which stored vectors are most similar to *this query*?" (one-to-many kNN);
this module answers "which stored vectors are near-duplicates of *each other*?"
(many-to-many). The latter is the first step of the planned Distill
de-duplication pipeline: detect near-duplicate IdeaBlocks, cluster them, then
merge each cluster into one canonical block (recording provenance via the
schema's ``parents`` / ``status`` / ``confidence`` lifecycle fields).

Like :class:`~sparksage.embed.store.InMemoryVectorStore`, this is pure Python
(no ``numpy`` / ``faiss``) -- ``O(n^2 * d)`` is fine for thousands of blocks.
For million-vector corpora the planned approximate index under ``[distill]``
(LSH candidate reduction + FAISS kNN) takes over; until then this exact
brute-force version keeps the core unit-testable with
:class:`~sparksage.embed.client.FakeEmbeddingClient` and zero dependencies.

Vectors are assumed L2-normalized (every
:class:`~sparksage.embed.client.EmbeddingClient` normalizes by default), so
cosine similarity reduces to a plain dot product -- the same convention the
store relies on. Feed it the ``{block_id: vector}`` mapping returned by
:meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for` (or
:meth:`~sparksage.embed.store.InMemoryVectorStore.vectors` from a persisted
store); clustering itself (connected components / Louvain) is intentionally
left to the future ``distill/`` package -- this module stops at the pair list.
"""

from __future__ import annotations

from dataclasses import dataclass


def _dot(a: list[float], b: list[float]) -> float:
    """Pure-Python dot product (no numpy needed)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class SimilarityPair:
    """A pair of near-duplicate block ids and their similarity score.

    Attributes
    ----------
    a, b:
        The two block ids, normalized so ``a <= b`` lexicographically. This
        guarantees each unordered pair appears at most once in a result set,
        independent of the input dict's iteration order.
    score:
        Cosine similarity between the two vectors -- a dot product, since every
        :class:`~sparksage.embed.client.EmbeddingClient` L2-normalizes. In
        ``[0, 1]`` for the near-duplicate use case (negative similarity means
        "opposite", not "duplicate").
    """

    a: str
    b: str
    score: float


def find_similar_pairs(
    vectors: dict[str, list[float]],
    *,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> list[SimilarityPair]:
    """Return all near-duplicate block pairs whose similarity >= ``threshold``.

    Computes the dot product between every unordered pair of vectors and keeps
    those at or above ``threshold``, sorted best (highest score) first. This is
    the exact-brute-force Distill candidate-pair step; results are fully
    deterministic: ties in score are broken by ``(a, b)`` lexicographically, so
    the same vector set always yields the same pair list regardless of dict
    insertion order.

    Parameters
    ----------
    vectors:
        A ``{block_id: vector}`` mapping -- typically the output of
        :meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for` or
        :meth:`~sparksage.embed.store.InMemoryVectorStore.vectors`. Every
        vector must have the same length.
    threshold:
        Minimum cosine similarity for a pair to count as a near-duplicate.
        Must be in ``[0, 1]``. The Distill pipeline conventionally starts at
        ``0.55`` and tightens by ``+0.01`` per iteration up to ``0.98``; the
        default ``0.5`` is a permissive baseline suitable for inspection.
    top_k:
        If given, return only the ``top_k`` highest-scoring pairs (after the
        ``threshold`` filter). ``None`` returns every pair above threshold.

    Returns
    -------
    list[SimilarityPair]
        Pairs with ``score >= threshold``, sorted by descending score then by
        ``(a, b)``. Each unordered pair appears at most once. Empty when fewer
        than two vectors are supplied or no pair clears ``threshold``.

    Raises
    ------
    TypeError
        If ``threshold`` / ``top_k`` have the wrong type.
    ValueError
        If ``threshold`` is outside ``[0, 1]``, ``top_k < 1``, or the vectors
        are not all the same length.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient, find_similar_pairs
    >>> embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
    >>> vectors = embedder.vectors_for(blocks)   # {block_id: vector}
    >>> for pair in find_similar_pairs(vectors, threshold=0.6):
    ...     print(f"{pair.score:.3f}  {pair.a} ~ {pair.b}")
    """

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a float")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an int or None")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

    ids = list(vectors.keys())
    if len(ids) < 2:
        return []

    dimension: int | None = None
    for bid, vec in vectors.items():
        if not isinstance(vec, list):
            raise TypeError(f"vector for {bid!r} must be a list")
        if dimension is None:
            dimension = len(vec)
        elif len(vec) != dimension:
            raise ValueError(
                f"inconsistent vector dimensions: {bid!r} has length "
                f"{len(vec)}; expected {dimension}"
            )

    assert dimension is not None  # narrowed: len(ids) >= 2 implies >= 1 entry

    pairs: list[SimilarityPair] = []
    n = len(ids)
    for i in range(n):
        id_i = ids[i]
        vec_i = vectors[id_i]
        for j in range(i + 1, n):
            id_j = ids[j]
            score = _dot(vec_i, vectors[id_j])
            if score >= threshold:
                a, b = (id_i, id_j) if id_i <= id_j else (id_j, id_i)
                pairs.append(SimilarityPair(a=a, b=b, score=score))

    pairs.sort(key=lambda p: (-p.score, p.a, p.b))
    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs
