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
For million-vector corpora an approximate candidate reducer (e.g. the LSH
reducer shipped under ``[distill]`` in :mod:`sparksage.distill.lsh`) takes
over: it cheaply proposes a small set of *candidate* pairs, which
:func:`find_similar_pairs` then verifies with exact dot products. Pass any
object implementing :class:`CandidateReducer` as ``candidate_reducer=`` -- the
exact brute-force path stays the default so the core stays unit-testable with
:class:`~sparksage.embed.client.FakeEmbeddingClient` and zero dependencies.

Vectors are assumed L2-normalized (every
:class:`~sparksage.embed.client.EmbeddingClient` normalizes by default), so
cosine similarity reduces to a plain dot product -- the same convention the
store relies on. Feed it the ``{block_id: vector}`` mapping returned by
:meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for` (or
:meth:`~sparksage.embed.store.InMemoryVectorStore.vectors` from a persisted
store); clustering itself (connected components / Louvain) is intentionally
left to the ``distill/`` package -- this module stops at the pair list.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.embed.store import dot


@runtime_checkable
class CandidateReducer(Protocol):
    """Cheap proposer of *candidate* near-duplicate id pairs.

    Implementations (e.g. :class:`~sparksage.distill.lsh.LSHCandidateReducer`)
    trade exactness for speed: instead of the ``O(n²·d)`` all-pairs scan they
    project vectors into hash buckets and only emit pairs that collide in at
    least one bucket. :func:`find_similar_pairs` still does the *exact* dot
    product on every candidate, so a reducer can only *drop* true duplicates
    (lowering recall), never invent false positives (precision stays 1.0).

    The protocol deliberately takes no ``threshold``: threshold tuning lives
    on the verification side; the reducer's job is purely to pre-filter the
    comparison set.
    """

    def candidate_pairs(
        self,
        vectors: dict[str, list[float]],
    ) -> Iterator[tuple[str, str]]:
        """Yield unordered ``(a, b)`` id pairs that *might* be near-duplicates.

        Each pair MUST be yielded with ``a <= b`` lexicographically so dedup is
        straightforward. Pairs whose ids are unknown to ``vectors`` are
        silently skipped by the verifier. Duplicates across calls/yields are
        allowed -- :func:`find_similar_pairs` dedupes internally.
        """
        ...


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
    candidate_reducer: CandidateReducer | None = None,
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
    candidate_reducer:
        Optional :class:`CandidateReducer` used to pre-filter the comparison
        set for million-vector corpora. When supplied, only the emitted
        candidate pairs are *verified* with exact dot products (the rest are
        assumed to be below ``threshold``). ``None`` (default) runs the exact
        ``O(n²·d)`` all-pairs scan. Precision is always 1.0: a reducer can
        only drop true positives (lowering recall), never introduce false
        positives, because every returned pair is still exact-verified.

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

    if candidate_reducer is not None:
        pairs = _verify_candidates(vectors, candidate_reducer.candidate_pairs(vectors), threshold)
    else:
        pairs = _brute_force_pairs(vectors, ids, threshold)

    pairs.sort(key=lambda p: (-p.score, p.a, p.b))
    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs


def _brute_force_pairs(
    vectors: dict[str, list[float]],
    ids: list[str],
    threshold: float,
) -> list[SimilarityPair]:
    """Exact all-pairs scan -- the default ``O(n²·d)`` path."""
    pairs: list[SimilarityPair] = []
    n = len(ids)
    for i in range(n):
        id_i = ids[i]
        vec_i = vectors[id_i]
        for j in range(i + 1, n):
            id_j = ids[j]
            score = dot(vec_i, vectors[id_j])
            if score >= threshold:
                a, b = (id_i, id_j) if id_i <= id_j else (id_j, id_i)
                pairs.append(SimilarityPair(a=a, b=b, score=score))
    return pairs


def _verify_candidates(
    vectors: dict[str, list[float]],
    candidates: Iterable[tuple[str, str]],
    threshold: float,
) -> list[SimilarityPair]:
    """Exact-verify a candidate stream and keep those whose dot product clears ``threshold``.

    Precision stays 1.0 -- only real dot products are emitted -- but recall
    depends entirely on the candidate generator. Pairs whose ids are not in
    ``vectors`` are skipped, and each unordered pair is verified at most once
    (the candidate stream may yield duplicates across hash tables).
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[SimilarityPair] = []
    for a, b in candidates:
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        if a not in vectors or b not in vectors:
            continue
        score = dot(vectors[a], vectors[b])
        if score >= threshold:
            pairs.append(SimilarityPair(a=a, b=b, score=score))
    return pairs
