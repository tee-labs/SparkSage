"""Near-duplicate clustering over embedding vectors (pure stdlib).

This is the second step of the Distill de-duplication pipeline -- it groups the
near-duplicate *pairs* produced by
:func:`~sparksage.embed.similarity.find_similar_pairs` into clusters of
mutually-similar blocks (connected components), ready to be merged into one
canonical block each by :mod:`sparksage.distill.merge`.

Design mirrors the rest of SparkSage: the core depends only on the existing
``{block_id: vector}`` contract and :class:`~sparksage.embed.similarity.SimilarityPair`,
so it is pure Python (no ``networkx`` / ``python-louvain`` -- those belong to the
future ``[distill]`` extra) and fully unit-testable offline.

Two algorithms ship:

* :class:`ConnectedComponentsBackend` -- the default. Union-find over the pair
  graph. ``O(n · α(n))``, deterministic, dependency-free. Used for corpora below
  the :data:`LOUVAIN_THRESHOLD` (or when the optional dependency is absent).
* :class:`LouvainClusteringBackend` -- modularity-based community detection for
  large corpora (>= :data:`LOUVAIN_THRESHOLD` nodes). The ``python-louvain``
  package is an *optional* dependency under ``[distill]``, imported lazily.

The :class:`DistillPipeline` never imports a concrete backend directly -- it
depends on the :class:`ClusteringBackend` protocol, so the choice is
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.embed.similarity import SimilarityPair, find_similar_pairs

#: Corpora at or above this size earn a Louvain pass (modularity community
#: detection) instead of plain connected components. The crossover is empirical:
#: below it the union-find components are exact and cheap; above it, the pair
#: graph gets dense enough that modularity finds tighter, non-transitive
#: clusters (avoiding chaining unrelated blocks through a few noisy edges).
LOUVAIN_THRESHOLD: int = 1000


# --------------------------------------------------------------------------- #
# Union-find (pure stdlib)
# --------------------------------------------------------------------------- #
class _UnionFind:
    """Weighted union-find with path compression (pure stdlib)."""

    def __init__(self, items: list[str]) -> None:
        self._parent: dict[str, str] = {item: item for item in items}
        self._rank: dict[str, int] = {item: 0 for item in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for item in self._parent:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        out = [sorted(g) for g in groups.values()]
        out.sort(key=lambda g: g[0])
        return out

    def component_count(self) -> int:
        return sum(1 for item, parent in self._parent.items() if item == parent)


@dataclass(frozen=True)
class Cluster:
    """A group of near-duplicate block ids ready to be merged into one canonical.

    Attributes
    ----------
    members:
        The block ids in this cluster (sorted lexicographically for determinism).
    confidence:
        Mean pairwise similarity within the cluster, in ``[0, 1]``. ``1.0`` for a
        singleton (no merge evidence). This is the value written to the canonical
        block's :attr:`~sparksage.schema.IdeaBlock.confidence` lifecycle field.
    """

    members: tuple[str, ...]
    confidence: float

    @property
    def size(self) -> int:
        return len(self.members)


# --------------------------------------------------------------------------- #
# Clustering backend protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class ClusteringBackend(Protocol):
    """Group a ``{block_id: vector}`` set into near-duplicate clusters.

    Any container that turns vectors at a given similarity ``threshold`` into a
    list of :class:`Cluster` s implements this -- the brute-force
    :class:`ConnectedComponentsBackend` in-process, or a Louvain / FAISS-backed
    backend in production. The :class:`~sparksage.distill.pipeline.DistillPipeline`
    depends on this protocol, never on a concrete implementation.
    """

    def cluster(
        self,
        vectors: dict[str, list[float]],
        *,
        threshold: float = 0.5,
    ) -> list[Cluster]:
        """Return every near-duplicate cluster at ``>= threshold`` similarity.

        Every input id appears in exactly one cluster; ids with no near-duplicate
        form singleton clusters. Clusters are returned in a deterministic order
        (by first member id).
        """
        ...


def _mean_pair_score(members: list[str], pairs: list[SimilarityPair]) -> float:
    """Mean similarity over the edges *within* ``members`` (1.0 if none)."""
    member_set = set(members)
    intra = [p.score for p in pairs if p.a in member_set and p.b in member_set]
    if not intra:
        return 1.0
    return sum(intra) / len(intra)


def _build_clusters(
    ids: list[str],
    pairs: list[SimilarityPair],
) -> list[Cluster]:
    """Turn a pair list into :class:`Cluster`s via union-find connected components."""
    uf = _UnionFind(ids)
    for pair in pairs:
        uf.union(pair.a, pair.b)
    clusters: list[Cluster] = []
    for group in uf.components():
        confidence = _mean_pair_score(group, pairs)
        clusters.append(Cluster(members=tuple(group), confidence=confidence))
    clusters.sort(key=lambda c: c.members[0])
    return clusters


class ConnectedComponentsBackend:
    """Default :class:`ClusteringBackend`: union-find connected components.

    Computes near-duplicate pairs with
    :func:`~sparksage.embed.similarity.find_similar_pairs` (exact brute force,
    pure stdlib) then unions their endpoints. Deterministic, dependency-free,
    ``O(n²·d)`` dominated by the pair scan -- fine for thousands of blocks. For
    larger corpora, swap in :class:`LouvainClusteringBackend` (or a future
    FAISS/LSH-backed candidate reducer under ``[distill]``).

    Parameters
    ----------
    min_cluster_size:
        Clusters strictly smaller than this are still returned (as singletons
        when no pair clears ``threshold``), but callers usually ignore clusters
        below 2. Kept as an attribute so introspection is cheap. Defaults to 1
        (return everything; the pipeline filters).
    """

    def __init__(self, min_cluster_size: int = 1) -> None:
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")
        self._min_cluster_size = min_cluster_size

    @property
    def min_cluster_size(self) -> int:
        return self._min_cluster_size

    def cluster(
        self,
        vectors: dict[str, list[float]],
        *,
        threshold: float = 0.5,
    ) -> list[Cluster]:
        ids = list(vectors.keys())
        if not ids:
            return []
        pairs = find_similar_pairs(vectors, threshold=threshold)
        return _build_clusters(ids, pairs)


class LouvainClusteringBackend:
    """:class:`ClusteringBackend` backed by modularity community detection.

    For large corpora (>= :data:`LOUVAIN_THRESHOLD` nodes) the pair graph becomes
    dense enough that plain connected components *chain* loosely-related blocks
    together via a few noisy edges. Louvain modularity maximization finds
    tighter, non-transitive communities instead.

    The ``python-louvain`` package (importable as ``community``) is an *optional*
    dependency -- install it with ``pip install 'sparksage[distill]'``. It is
    imported lazily on the first :meth:`cluster` call so the module loads
    cleanly without it.

    Parameters
    ----------
    resolution:
        Louvain resolution parameter (higher -> more, smaller communities).
        Defaults to ``1.0`` (standard modularity).
    """

    def __init__(self, *, resolution: float = 1.0) -> None:
        self._resolution = resolution

    def _import_graph(self) -> tuple[object, object]:
        try:
            import community as community_louvain
            import networkx as nx  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "LouvainClusteringBackend requires 'networkx' and "
                "'python-louvain'. Install them with: "
                "pip install 'sparksage[distill]'"
            ) from exc
        import networkx as nx

        return nx, community_louvain

    def cluster(
        self,
        vectors: dict[str, list[float]],
        *,
        threshold: float = 0.5,
    ) -> list[Cluster]:
        nx, community_louvain = self._import_graph()  # pragma: no cover
        ids = list(vectors.keys())
        if not ids:
            return []
        pairs = find_similar_pairs(vectors, threshold=threshold)

        graph = nx.Graph()  # type: ignore[attr-defined]
        graph.add_nodes_from(ids)
        for pair in pairs:
            graph.add_edge(pair.a, pair.b, weight=pair.score)

        if graph.number_of_edges() == 0:
            return [Cluster(members=(bid,), confidence=1.0) for bid in ids]

        partition = community_louvain.best_partition(  # type: ignore[attr-defined]
            graph, resolution=self._resolution, random_state=42
        )
        groups: dict[int, list[str]] = {}
        for bid, comm in partition.items():
            groups.setdefault(comm, []).append(bid)

        clusters: list[Cluster] = []
        for members in groups.values():
            members_sorted = sorted(members)
            clusters.append(
                Cluster(
                    members=tuple(members_sorted),
                    confidence=_mean_pair_score(members_sorted, pairs),
                )
            )
        clusters.sort(key=lambda c: c.members[0])
        return clusters


def select_clustering_backend(
    n_vectors: int,
    *,
    louvain_threshold: int = LOUVAIN_THRESHOLD,
    prefer_louvain: bool = True,
) -> ClusteringBackend:
    """Pick a sensible default backend for a corpus of ``n_vectors`` blocks.

    Returns :class:`LouvainClusteringBackend` when the corpus is large enough
    *and* the optional dependency is importable *and* ``prefer_louvain``;
    otherwise :class:`ConnectedComponentsBackend` (the dependency-free default).
    The dependency probe is cached so it is cheap to call per run.
    """
    if prefer_louvain and n_vectors >= louvain_threshold and _has_louvain():
        return LouvainClusteringBackend()
    return ConnectedComponentsBackend()


@dataclass
class _Probe:
    cached: bool | None = None


_PROBE = _Probe()


def _has_louvain() -> bool:
    if _PROBE.cached is None:
        try:
            import community  # noqa: F401
            import networkx  # noqa: F401
        except ImportError:
            _PROBE.cached = False
        else:
            _PROBE.cached = True
    return _PROBE.cached


# --------------------------------------------------------------------------- #
# Hierarchical merge support: partition a large cluster into tighter sub-groups
# --------------------------------------------------------------------------- #
def partition_by_strongest_edges(
    members: list[str],
    pairs: list[SimilarityPair],
    n_groups: int,
) -> list[list[str]]:
    """Split ``members`` into at most ``n_groups`` sub-groups by strongest edges.

    Used by the hierarchical merge step of the Distill pipeline: a cluster larger
    than the per-call LLM budget is recursively partitioned by unioning its
    highest-similarity edges first, stopping once ``n_groups`` components remain.
    This breaks a loosely-chained cluster into the tightest natural sub-clusters
    without any extra embedding calls.

    Guarantees:

    * every member appears in exactly one returned group;
    * the number of groups is ``<= max(1, min(n_groups, len(members)))`` -- when
      the strongest edges alone cannot reduce that far (a loosely-connected
      cluster), the remainder is split evenly so the bound always holds;
    * groups are sorted (within by id, overall by first id) for determinism;
    * when ``n_groups >= len(members)`` each member is its own group.

    Parameters
    ----------
    members:
        Block ids that must all appear in the result.
    pairs:
        Candidate edges (only those with both endpoints in ``members`` are used).
    n_groups:
        Target number of sub-groups. Clamped to ``[1, len(members)]``.
    """
    if not members:
        return []
    n = len(members)
    target = max(1, min(n_groups, n))

    member_set = set(members)
    intra = [p for p in pairs if p.a in member_set and p.b in member_set]
    intra.sort(key=lambda p: (-p.score, p.a, p.b))

    uf = _UnionFind(members)
    for pair in intra:
        if uf.component_count() <= target:
            break
        uf.union(pair.a, pair.b)

    components = uf.components()
    if len(components) <= target:
        components.sort(key=lambda g: g[0])
        return components

    # Too few edges to reach ``target`` components via unioning alone: the
    # cluster is loosely connected. Fall back to an even slice of the sorted
    # members so the group-count bound still holds (and the hierarchical merge
    # always strictly reduces a large cluster -- never recurses forever).
    return _even_slice(sorted(member_set), target)


def _even_slice(members: list[str], n_groups: int) -> list[list[str]]:
    """Split ``members`` (already sorted) into ``n_groups`` contiguous groups."""
    n = len(members)
    n_groups = max(1, min(n_groups, n))
    base = n // n_groups
    rem = n % n_groups
    groups: list[list[str]] = []
    i = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        groups.append(members[i : i + size])
        i += size
    return groups


__all__ = [
    "LOUVAIN_THRESHOLD",
    "Cluster",
    "ClusteringBackend",
    "ConnectedComponentsBackend",
    "LouvainClusteringBackend",
    "partition_by_strongest_edges",
    "select_clustering_backend",
]
