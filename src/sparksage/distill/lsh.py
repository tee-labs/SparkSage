"""LSH candidate reduction: the ``[distill]`` accelerator for million-vector dedup.

Replaces the ``O(n^2 * d)`` brute-force scan inside
:func:`~sparksage.embed.similarity.find_similar_pairs` with random-hyperplane
LSH: project each vector onto ``k`` random hyperplanes per table, hash the
sign pattern into a ``k``-bit signature, and only *verify* pairs that collide
in at least one of ``L`` independent tables. For two vectors with angle
``theta`` the per-table collision probability is ``(1 - theta/pi)^k``; ``L``
tables give recall ``1 - (1 - (1 - theta/pi)^k)^L`` (OR-amplification across
tables, AND-amplification within a table).

Pure stdlib -- hyperplanes are Gaussian vectors drawn from
:class:`random.Random`, hashing is the sign of the dot product, no ``numpy``.
This matches the rest of the Distill core: the reducer is fully unit-testable
offline with :class:`~sparksage.embed.FakeEmbeddingClient` and zero optional
dependencies, exactly like :class:`~sparksage.distill.ConnectedComponentsBackend`.
Users who need numpy-level throughput on truly massive corpora may swap in
their own reducer that implements
:class:`~sparksage.embed.similarity.CandidateReducer`.

The reducer implements :class:`~sparksage.embed.similarity.CandidateReducer`;
:func:`~sparksage.embed.similarity.find_similar_pairs` is still responsible for
the *exact* dot-product verification, so **precision stays 1.0** -- the reducer
can only drop true duplicates (lowering recall), never invent false positives.

Default parameters (``num_hyperplanes=6``, ``num_tables=20``, ``seed=42``) give
~89% recall at cosine 0.55 (the Distill starting threshold) and 97%+ across the
tightened regime, while cutting comparisons several-fold on corpora where the
near-duplicate fraction is small (the regime where LSH earns its keep). Tune
``num_hyperplanes`` down for higher recall at the cost of more candidates; tune
``num_tables`` up for higher recall at linear memory/time cost. Use
:meth:`LSHCandidateReducer.theoretical_recall` to check the trade-off up front.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

#: Corpora at or above this size earn an LSH pass; below it the exact
#: ``O(n^2 * d)`` brute force is faster (LSH's hashing overhead -- ``n * L * k
#: * d`` -- exceeds the savings until a few thousand vectors). Mirrors
#: :data:`~sparksage.distill.cluster.LOUVAIN_THRESHOLD`'s role for clustering
#: backend selection.
LSH_ACTIVATION_THRESHOLD: int = 5000

#: Default number of random hyperplanes per hash table (the AND-amplification
#: width). Higher -> tighter buckets -> more compression -> lower recall at
#: any fixed similarity. Default ``6`` targets ~89% recall at cosine 0.55.
DEFAULT_NUM_HYPERPLANES: int = 6

#: Default number of independent hash tables (the OR-amplification breadth).
#: Higher -> higher recall at linear memory/time cost. Default ``20``.
DEFAULT_NUM_TABLES: int = 20

#: Default seed for the random hyperplanes. Deterministic so the same corpus
#: + reducer yields the same candidate set across runs.
DEFAULT_SEED: int = 42


class LSHCandidateReducer:
    """Random-hyperplane LSH candidate reducer for near-duplicate detection.

    Implements :class:`~sparksage.embed.similarity.CandidateReducer`. Pure
    stdlib: hyperplanes are Gaussian vectors drawn from :class:`random.Random`,
    hashing is the sign of the dot product, no ``numpy`` needed.

    Parameters
    ----------
    num_hyperplanes:
        Per-table AND-amplification width (default
        :data:`DEFAULT_NUM_HYPERPLANES`).
    num_tables:
        Number of independent tables (OR-amplification breadth; default
        :data:`DEFAULT_NUM_TABLES`).
    seed:
        Seed for hyperplane generation. Deterministic per seed so the same
        corpus + reducer always produces the same candidate set.

    Notes
    -----
    Recall at cosine similarity ``s`` (angle ``theta = arccos(s)``):

    .. math::

        R(s) = 1 - \\left(1 - \\left(\\frac{\\pi - \\arccos(s)}{\\pi}\\right)^{k}\\right)^{L}

    Precision is always ``1.0`` because
    :func:`~sparksage.embed.similarity.find_similar_pairs` exact-verifies every
    candidate via dot product. The reducer can only *lower recall*, never
    produce false positives.

    Examples
    --------
    >>> from sparksage import LSHCandidateReducer, find_similar_pairs
    >>> reducer = LSHCandidateReducer()                       # doctest: +SKIP
    >>> pairs = find_similar_pairs(                           # doctest: +SKIP
    ...     vectors, threshold=0.55, candidate_reducer=reducer,
    ... )
    >>> reducer.theoretical_recall(0.55)                      # ~0.89
    0.88...
    """

    def __init__(
        self,
        *,
        num_hyperplanes: int = DEFAULT_NUM_HYPERPLANES,
        num_tables: int = DEFAULT_NUM_TABLES,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if isinstance(num_hyperplanes, bool) or not isinstance(num_hyperplanes, int):
            raise TypeError("num_hyperplanes must be an int")
        if num_hyperplanes < 1:
            raise ValueError("num_hyperplanes must be >= 1")
        if isinstance(num_tables, bool) or not isinstance(num_tables, int):
            raise TypeError("num_tables must be an int")
        if num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self._num_hyperplanes = num_hyperplanes
        self._num_tables = num_tables
        self._seed = seed
        # Lazily built per-dimension hyperplane tables. None until the first
        # candidate_pairs() call fixes the dimension; rebuilt if a later call
        # sees a different dimension (defensive -- the pipeline feeds the same
        # embedder throughout, so this is rare).
        self._tables: list[list[list[list[float]]]] | None = None
        self._dimension: int | None = None

    @property
    def num_hyperplanes(self) -> int:
        return self._num_hyperplanes

    @property
    def num_tables(self) -> int:
        return self._num_tables

    @property
    def seed(self) -> int:
        return self._seed

    def theoretical_recall(self, similarity: float) -> float:
        """Closed-form collision recall at a given cosine ``similarity``.

        Handy for tuning ``num_hyperplanes`` / ``num_tables`` up front without
        running the reducer. ``similarity`` must be in ``[-1, 1]``.
        """
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            raise TypeError("similarity must be a float")
        similarity = float(similarity)
        if not -1.0 <= similarity <= 1.0:
            raise ValueError("similarity must be in [-1, 1]")
        theta = math.acos(max(-1.0, min(1.0, similarity)))
        p_table = (1.0 - theta / math.pi) ** self._num_hyperplanes
        return 1.0 - (1.0 - p_table) ** self._num_tables

    # ------------------------------------------------------------------ #
    # CandidateReducer protocol
    # ------------------------------------------------------------------ #
    def candidate_pairs(
        self,
        vectors: dict[str, list[float]],
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(a, b)`` candidate id pairs with ``a <= b``, deduped across tables.

        Each unordered pair is yielded at most once even though it may collide
        in several tables. Pairs are streamed lazily so memory scales with the
        bucket sizes, not the full candidate set.
        """
        if not vectors:
            return
        sample = next(iter(vectors.values()))
        dim = len(sample)
        if dim < 1:
            return
        if self._tables is None or self._dimension != dim:
            self._build_tables(dim)

        seen: set[tuple[str, str]] = set()
        assert self._tables is not None
        for table in self._tables:
            buckets: dict[tuple[bool, ...], list[str]] = {}
            for bid, vec in vectors.items():
                sig = self._signature(vec, table)
                buckets.setdefault(sig, []).append(bid)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                members_sorted = sorted(members)
                n = len(members_sorted)
                for i in range(n):
                    a = members_sorted[i]
                    for j in range(i + 1, n):
                        b = members_sorted[j]
                        key = (a, b)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield key

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_tables(self, dimension: int) -> None:
        """Draw ``num_tables`` independent tables of ``num_hyperplanes`` Gaussian hyperplanes.

        A fresh :class:`random.Random` seeded from ``self._seed`` keeps the
        reducer deterministic: the same ``(dimension, num_hyperplanes,
        num_tables, seed)`` always produces the same hyperplanes, hence the
        same candidate set for the same corpus.
        """
        rng = random.Random(self._seed)
        tables: list[list[list[list[float]]]] = []
        for _ in range(self._num_tables):
            table: list[list[float]] = []
            for _ in range(self._num_hyperplanes):
                table.append([rng.gauss(0.0, 1.0) for _ in range(dimension)])
            tables.append(table)
        self._tables = tables
        self._dimension = dimension

    @staticmethod
    def _signature(vec: list[float], table: list[list[float]]) -> tuple[bool, ...]:
        """The ``k``-bit LSH signature: one bit per hyperplane (sign of dot product).

        ``strict=True`` on ``zip`` defends against a hyperplane whose dimension
        drifted from the vectors' -- should never happen given
        :meth:`_build_tables`, but the failure mode (silent truncation) would
        be subtle.
        """
        return tuple(
            sum(v * p for v, p in zip(vec, plane, strict=True)) > 0.0
            for plane in table
        )


def select_candidate_reducer(
    n_vectors: int,
    *,
    activation_threshold: int = LSH_ACTIVATION_THRESHOLD,
    prefer_lsh: bool = True,
    **lsh_kwargs: object,
) -> LSHCandidateReducer | None:
    """Pick a sensible candidate reducer for a corpus of ``n_vectors`` blocks.

    Returns an :class:`LSHCandidateReducer` when the corpus is large enough
    (``>= activation_threshold``, default :data:`LSH_ACTIVATION_THRESHOLD`) and
    ``prefer_lsh`` is true; otherwise ``None``, meaning
    :func:`~sparksage.embed.similarity.find_similar_pairs` will run its exact
    brute-force path. The threshold is empirical: below it the LSH hashing
    overhead (``n * L * k * d``) exceeds the savings from skipping the
    ``n^2 / 2`` dot-product comparisons.

    Any extra ``lsh_kwargs`` are forwarded to :class:`LSHCandidateReducer`
    (e.g. ``num_hyperplanes=`` / ``num_tables=`` / ``seed=``).
    """
    if prefer_lsh and n_vectors >= activation_threshold:
        return LSHCandidateReducer(**lsh_kwargs)  # type: ignore[arg-type]
    return None


__all__ = [
    "DEFAULT_NUM_HYPERPLANES",
    "DEFAULT_NUM_TABLES",
    "DEFAULT_SEED",
    "LSH_ACTIVATION_THRESHOLD",
    "LSHCandidateReducer",
    "select_candidate_reducer",
]
