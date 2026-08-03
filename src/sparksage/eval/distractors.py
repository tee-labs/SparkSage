"""Adversarial distractor injection: measure retrieval robustness under traps.

This is the robustness counterpart to :mod:`sparksage.bench` (which measures
retrieval quality on a clean corpus) and :mod:`sparksage.eval.evaluator` (which
measures end-to-end answer correctness). It answers a different question:
*can the retriever resist semantically-similar-but-wrong distractors?*

The paper ("BM25 Wins at Scale") identified **traps** -- content that is
dense-similar to a query but factually wrong -- as the key weakness of dense
retrieval: dense vectors surface plausible-looking false positives. This module
makes that failure mode *measurable*:

* :class:`DistractorInjector` generates **trap blocks** -- blocks that mimic a
  target's name + ``critical_question`` (so dense retrieval finds them) but
  carry a **wrong** ``trusted_answer`` borrowed from a semantically similar
  donor. No LLM is needed: the embedder finds the donor, the swap is mechanical.
* :class:`RobustnessEvaluator` injects traps into the corpus, builds both the
  IdeaBlock and the naive-chunk index (the same A/B comparison
  :class:`~sparksage.bench.BenchmarkRunner` uses), queries each target's
  ``critical_question``, and measures how often traps contaminate the top-k.
* :class:`RobustnessReport` reports the **true-hit rate** (does the right block
  surface?) vs the **trap-contamination rate** (does any wrong block surface?)
  for both strategies, so the IdeaBlock ``trusted_answer`` dividend is visible
  as lower contamination.

This directly quantifies IdeaBlock's design advantage: the curated
``trusted_answer`` is embedded as part of the block, so a trap carrying a wrong
answer produces a measurably different vector than the true block -- something
naive chunks (which may split the answer across boundaries) cannot guarantee.

The evaluator is pure stdlib beyond :class:`~sparksage.embed.BlockEmbedder`,
:class:`~sparksage.embed.store.InMemoryVectorStore`, and
:class:`~sparksage.bench.RecursiveCharSplitter`, so it runs offline with
:class:`~sparksage.embed.FakeEmbeddingClient`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sparksage.bench.baselines import RecursiveCharSplitter
from sparksage.bench.report import _safe_ratio
from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.store import InMemoryVectorStore
from sparksage.schema.ideablock import IdeaBlock

#: Default ``k`` for the robustness top-k contamination window.
DEFAULT_ROBUSTNESS_K = 5

#: Default number of trap blocks generated per target block.
DEFAULT_TRAPS_PER_BLOCK = 1


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass
class TrapRecord:
    """Records one generated trap block and its provenance.

    Attributes
    ----------
    trap_block:
        The generated :class:`~sparksage.schema.IdeaBlock` (target's question,
        donor's answer).
    target_block_id:
        Id of the block whose ``critical_question`` the trap mimics.
    donor_block_id:
        Id of the block whose ``trusted_answer`` was swapped in.
    similarity:
        Target-donor embedding cosine similarity (the higher, the harder the
        trap -- the donor was a close neighbor).
    donor_rank:
        Zero-based rank of the donor among the target's neighbours (0 = closest
        non-identical donor).
    """

    trap_block: IdeaBlock
    target_block_id: str
    donor_block_id: str
    similarity: float = 0.0
    donor_rank: int = 0

    @property
    def trap_id(self) -> str:
        return str(self.trap_block.id)


class DistractorInjector:
    """Generate adversarial trap blocks from a real corpus (no LLM needed).

    For each target block, finds the most semantically similar *donor* (via the
    embedder's cosine similarity), then creates a trap that copies the target's
    ``name`` + ``critical_question`` but swaps in the donor's
    ``trusted_answer``. The trap's ``embedding_text`` is therefore ~2/3
    identical to the target's -- a strong dense distractor that differs only in
    the answer portion. Donors with identical answers are skipped (a trivial
    trap is no trap at all).

    Parameters
    ----------
    embedder:
        A :class:`~sparksage.embed.BlockEmbedder` used to find donor candidates.
    min_similarity:
        Minimum target-donor cosine to accept a donor (default ``0.0`` = accept
        any non-identical donor). Raise this to generate only hard traps.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient   # doctest: +SKIP
    >>> injector = DistractorInjector(                            # doctest: +SKIP
    ...     BlockEmbedder(FakeEmbeddingClient()),
    ... )
    >>> traps = injector.generate_traps(blocks)                   # doctest: +SKIP
    """

    def __init__(
        self,
        embedder: BlockEmbedder,
        *,
        min_similarity: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._min_similarity = float(min_similarity)

    def generate_traps(
        self,
        blocks: list[IdeaBlock],
        *,
        traps_per_block: int = DEFAULT_TRAPS_PER_BLOCK,
    ) -> list[TrapRecord]:
        """Return up to ``traps_per_block`` traps per block.

        Blocks with fewer than ``traps_per_block`` qualifying donors (dissimilar
        answer, above ``min_similarity``) produce fewer traps. Returns an empty
        list when the corpus has fewer than 2 blocks.
        """
        if traps_per_block < 1:
            raise ValueError("traps_per_block must be >= 1")
        n = len(blocks)
        if n < 2:
            return []
        vectors = self._embedder.embed_texts(
            [b.embedding_text for b in blocks]
        )
        traps: list[TrapRecord] = []
        for i, target in enumerate(blocks):
            target_vec = vectors[i]
            scored: list[tuple[float, int, IdeaBlock]] = []
            for j, donor in enumerate(blocks):
                if j == i:
                    continue
                if donor.trusted_answer.strip() == target.trusted_answer.strip():
                    continue
                sim = _dot(target_vec, vectors[j])
                if sim < self._min_similarity:
                    continue
                scored.append((sim, j, donor))
            scored.sort(key=lambda t: t[0], reverse=True)
            for rank, (sim, _, donor) in enumerate(scored[:traps_per_block]):
                trap = IdeaBlock(
                    name=target.name,
                    critical_question=target.critical_question,
                    trusted_answer=donor.trusted_answer,
                    keywords=list(target.keywords),
                    tags=list(target.tags),
                    language=target.language,
                )
                traps.append(
                    TrapRecord(
                        trap_block=trap,
                        target_block_id=str(target.id),
                        donor_block_id=str(donor.id),
                        similarity=sim,
                        donor_rank=rank,
                    )
                )
        return traps

    def build_adversarial_corpus(
        self,
        blocks: list[IdeaBlock],
        *,
        traps_per_block: int = DEFAULT_TRAPS_PER_BLOCK,
    ) -> tuple[list[IdeaBlock], list[TrapRecord]]:
        """Return ``(true_blocks + trap_blocks, trap_records)``.

        The returned corpus is the original blocks followed by the generated
        trap blocks -- ready to be indexed by the evaluator. The original block
        list is not mutated.
        """
        traps = self.generate_traps(blocks, traps_per_block=traps_per_block)
        corpus = list(blocks) + [t.trap_block for t in traps]
        return corpus, traps


@dataclass
class RobustnessCaseResult:
    """The scored outcome of one adversarial retrieval query.

    Attributes
    ----------
    query:
        The query text (the target block's ``critical_question``).
    target_block_id:
        The ground-truth block id for this query.
    trap_ids:
        Ids of trap blocks targeting this query's block.
    retrieved_ids:
        Block (or chunk-derived) ids retrieved, best first, capped at ``k``.
    true_hit:
        Whether the true target appeared in the top-``k``.
    trap_hit:
        Whether any trap appeared in the top-``k``.
    first_trap_rank:
        One-based rank of the first trap in the ranking, or ``None``.
    """

    query: str
    target_block_id: str
    trap_ids: set[str] = field(default_factory=set)
    retrieved_ids: list[str] = field(default_factory=list)
    true_hit: bool = False
    trap_hit: bool = False
    first_trap_rank: int | None = None


@dataclass
class StrategyRobustness:
    """One strategy's adversarial robustness aggregate.

    Attributes
    ----------
    name:
        Strategy name ("IdeaBlock" / "Naive chunks").
    true_hit_rate:
        Fraction of queries where the true block surfaced in top-``k``.
    trap_contamination_rate:
        Fraction of queries where any trap surfaced in top-``k``. Lower is
        better -- the strategy resisted the distractor.
    mean_first_trap_rank:
        Mean rank of the first trap across contaminated queries, or ``None``
        when no query was contaminated.
    case_results:
        Per-query :class:`RobustnessCaseResult` list (for slicing / debugging).
    """

    name: str
    true_hit_rate: float = 0.0
    trap_contamination_rate: float = 0.0
    mean_first_trap_rank: float | None = None
    case_results: list[RobustnessCaseResult] = field(default_factory=list)


@dataclass
class RobustnessReport:
    """Side-by-side adversarial robustness comparison.

    Attributes
    ----------
    case_count:
        Number of queries evaluated (one per target block).
    trap_count:
        Total trap blocks injected into the corpus.
    k:
        The top-``k`` contamination window.
    ideablock, baseline:
        :class:`StrategyRobustness` for each strategy.
    config:
        Free-form config snapshot.
    generated_at:
        UTC timestamp the report was produced.
    """

    case_count: int = 0
    trap_count: int = 0
    k: int = DEFAULT_ROBUSTNESS_K
    ideablock: StrategyRobustness = field(default_factory=StrategyRobustness)
    baseline: StrategyRobustness = field(default_factory=StrategyRobustness)
    config: dict[str, object] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def trap_resistance_improvement(self) -> float:
        """How many times *lower* the IdeaBlock contamination is vs baseline.

        ``> 1.0`` means IdeaBlock resists traps better. Computed as
        ``baseline_contamination / ideablock_contamination``; returns ``0.0``
        when neither strategy was contaminated (no traps surfaced at all).
        """
        ib = self.ideablock.trap_contamination_rate
        base = self.baseline.trap_contamination_rate
        if ib == 0.0:
            return float("inf") if base > 0.0 else 0.0
        return _safe_ratio(base, ib)

    @property
    def mean_trap_similarity(self) -> float:
        """Mean target-donor similarity of injected traps (trap hardness)."""
        sims = self.config.get("trap_similarities")
        if not isinstance(sims, list) or not sims:
            return 0.0
        return sum(sims) / len(sims)

    def summary(self) -> str:
        """One-paragraph summary of the adversarial comparison."""
        if self.case_count == 0:
            return "No cases evaluated."
        ib = self.ideablock
        base = self.baseline
        improvement = self.trap_resistance_improvement
        imp_text = (
            f"{improvement:.2f}x better"
            if improvement != float("inf")
            else "perfect (no traps surfaced)"
        )
        return (
            f"Adversarial robustness over {self.case_count} queries "
            f"({self.trap_count} traps injected, top-{self.k}): "
            f"IdeaBlock true-hit {ib.true_hit_rate:.1%} / "
            f"contamination {ib.trap_contamination_rate:.1%} vs "
            f"naive true-hit {base.true_hit_rate:.1%} / "
            f"contamination {base.trap_contamination_rate:.1%} "
            f"(IdeaBlock trap resistance: {imp_text})."
        )

    def to_dict(self) -> dict[str, object]:
        """Plain (JSON-safe) dict view of the report."""
        return {
            "case_count": self.case_count,
            "trap_count": self.trap_count,
            "k": self.k,
            "config": {k: v for k, v in self.config.items() if k != "trap_similarities"},
            "mean_trap_similarity": self.mean_trap_similarity,
            "generated_at": self.generated_at.isoformat(),
            "strategies": {
                "ideablock": _strategy_dict(self.ideablock),
                "baseline": _strategy_dict(self.baseline),
            },
        }

    def to_html(self) -> str:
        """Render the report as a self-contained HTML document."""
        return _render_robustness_html(self)


def _strategy_dict(s: StrategyRobustness) -> dict[str, object]:
    return {
        "name": s.name,
        "true_hit_rate": s.true_hit_rate,
        "trap_contamination_rate": s.trap_contamination_rate,
        "mean_first_trap_rank": s.mean_first_trap_rank,
    }


class RobustnessEvaluator:
    """Measure retrieval robustness against adversarial distractors.

    Generates trap blocks via :class:`DistractorInjector`, injects them into the
    corpus, builds both the IdeaBlock and naive-chunk indexes, queries each
    target's ``critical_question``, and measures how often traps contaminate
    the top-k for each strategy. The comparison is fair by construction: same
    embedder, same queries, same traps -- only the chunking strategy differs.

    Parameters
    ----------
    embedder:
        A :class:`~sparksage.embed.BlockEmbedder` for indexing + donor search.
    splitter:
        Baseline chunker (default :class:`RecursiveCharSplitter`). Only used
        for the naive strategy.
    k_values:
        Unused here (kept for API parity with the bench runner); the
        contamination window is controlled by the ``k`` parameter of
        :meth:`evaluate`.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient   # doctest: +SKIP
    >>> evaluator = RobustnessEvaluator(                          # doctest: +SKIP
    ...     BlockEmbedder(FakeEmbeddingClient()),
    ... )
    >>> report = evaluator.evaluate(blocks, k=5)                  # doctest: +SKIP
    >>> report.ideablock.trap_contamination_rate                   # doctest: +SKIP
    """

    def __init__(
        self,
        embedder: BlockEmbedder,
        *,
        splitter: RecursiveCharSplitter | None = None,
    ) -> None:
        self._embedder = embedder
        self._splitter = splitter if splitter is not None else RecursiveCharSplitter()
        self._injector = DistractorInjector(embedder)

    @property
    def injector(self) -> DistractorInjector:
        return self._injector

    def evaluate(
        self,
        blocks: list[IdeaBlock],
        *,
        k: int = DEFAULT_ROBUSTNESS_K,
        traps_per_block: int = DEFAULT_TRAPS_PER_BLOCK,
    ) -> RobustnessReport:
        """Run the adversarial robustness comparison and return a report.

        Parameters
        ----------
        blocks:
            The real corpus (true blocks). Must have at least 2 blocks for trap
            generation.
        k:
            Top-k contamination window (default ``5``).
        traps_per_block:
            Traps generated per target (default ``1``).
        """
        if not blocks:
            raise ValueError("evaluate() requires at least one IdeaBlock")
        if k < 1:
            raise ValueError("k must be >= 1")
        if len(blocks) < 2:
            return self._empty_report(blocks, k)

        traps = self._injector.generate_traps(
            blocks, traps_per_block=traps_per_block
        )
        if not traps:
            return self._empty_report(blocks, k)

        trap_blocks = [t.trap_block for t in traps]
        corpus = list(blocks) + trap_blocks
        dimension = self._embedder.dimension

        target_to_traps: dict[str, set[str]] = {}
        for rec in traps:
            target_to_traps.setdefault(rec.target_block_id, set()).add(rec.trap_id)

        ib_cases = self._evaluate_ideablock_strategy(
            corpus, blocks, target_to_traps, dimension, k
        )
        base_cases = self._evaluate_baseline_strategy(
            corpus, blocks, trap_blocks, target_to_traps, dimension, k
        )

        ib_strategy = _aggregate("IdeaBlock", ib_cases)
        base_strategy = _aggregate("Naive chunks", base_cases)

        config: dict[str, object] = {
            "k": k,
            "traps_per_block": traps_per_block,
            "embedding_dimension": dimension,
            "splitter": type(self._splitter).__name__,
            "chunk_size": self._splitter.chunk_size,
            "chunk_overlap": self._splitter.chunk_overlap,
            "trap_similarities": [round(t.similarity, 6) for t in traps],
        }

        return RobustnessReport(
            case_count=len(blocks),
            trap_count=len(traps),
            k=k,
            ideablock=ib_strategy,
            baseline=base_strategy,
            config=config,
        )

    def _empty_report(
        self, blocks: list[IdeaBlock], k: int
    ) -> RobustnessReport:
        """Report for a corpus too small to generate traps."""
        return RobustnessReport(
            case_count=len(blocks),
            trap_count=0,
            k=k,
            ideablock=StrategyRobustness(name="IdeaBlock"),
            baseline=StrategyRobustness(name="Naive chunks"),
            config={"note": "corpus too small for trap generation (< 2 blocks)"},
        )

    def _evaluate_ideablock_strategy(
        self,
        corpus: list[IdeaBlock],
        query_blocks: list[IdeaBlock],
        target_to_traps: dict[str, set[str]],
        dimension: int,
        k: int,
    ) -> list[RobustnessCaseResult]:
        store = InMemoryVectorStore(dimension=dimension)
        vectors = self._embedder.vectors_for(corpus)
        store.add_many(vectors)
        query_vecs = self._embedder.embed_texts(
            [b.critical_question for b in query_blocks]
        )
        cases: list[RobustnessCaseResult] = []
        for qb, qv in zip(query_blocks, query_vecs, strict=True):
            hits = store.search(qv, k=k)
            retrieved = [h.block_id for h in hits]
            tid = str(qb.id)
            trap_ids = target_to_traps.get(tid, set())
            cases.append(
                _score_case(qb.critical_question, tid, trap_ids, retrieved)
            )
        return cases

    def _evaluate_baseline_strategy(
        self,
        corpus: list[IdeaBlock],
        query_blocks: list[IdeaBlock],
        trap_blocks: list[IdeaBlock],
        target_to_traps: dict[str, set[str]],
        dimension: int,
        k: int,
    ) -> list[RobustnessCaseResult]:
        chunks = self._splitter.split_blocks(corpus)
        store = InMemoryVectorStore(dimension=dimension)
        if chunks:
            chunk_vecs = self._embedder.embed_texts([c.text for c in chunks])
            for chunk, vec in zip(chunks, chunk_vecs, strict=True):
                store.add(chunk.id, list(vec))

        block_to_chunks: dict[str, set[str]] = {str(b.id): set() for b in corpus}
        for chunk in chunks:
            if chunk.source_ref_id and chunk.source_ref_id in block_to_chunks:
                block_to_chunks[chunk.source_ref_id].add(chunk.id)

        query_vecs = self._embedder.embed_texts(
            [b.critical_question for b in query_blocks]
        )
        cases: list[RobustnessCaseResult] = []
        for qb, qv in zip(query_blocks, query_vecs, strict=True):
            hits = store.search(qv, k=k)
            retrieved_chunk_ids = [h.block_id for h in hits]
            tid = str(qb.id)
            true_chunk_ids = block_to_chunks.get(tid, set())
            trap_block_ids = target_to_traps.get(tid, set())
            trap_chunk_ids: set[str] = set()
            for trap_bid in trap_block_ids:
                trap_chunk_ids |= block_to_chunks.get(trap_bid, set())
            true_hit = bool(true_chunk_ids) and any(
                cid in true_chunk_ids for cid in retrieved_chunk_ids
            )
            trap_hit = bool(trap_chunk_ids) and any(
                cid in trap_chunk_ids for cid in retrieved_chunk_ids
            )
            first_trap_rank: int | None = None
            for rank, cid in enumerate(retrieved_chunk_ids, start=1):
                if cid in trap_chunk_ids:
                    first_trap_rank = rank
                    break
            cases.append(
                RobustnessCaseResult(
                    query=qb.critical_question,
                    target_block_id=tid,
                    trap_ids=trap_chunk_ids,
                    retrieved_ids=retrieved_chunk_ids,
                    true_hit=true_hit,
                    trap_hit=trap_hit,
                    first_trap_rank=first_trap_rank,
                )
            )
        return cases


def _score_case(
    query: str,
    target_id: str,
    trap_ids: set[str],
    retrieved: list[str],
) -> RobustnessCaseResult:
    true_hit = target_id in retrieved
    trap_hit = bool(trap_ids) and any(rid in trap_ids for rid in retrieved)
    first_trap_rank: int | None = None
    for rank, rid in enumerate(retrieved, start=1):
        if rid in trap_ids:
            first_trap_rank = rank
            break
    return RobustnessCaseResult(
        query=query,
        target_block_id=target_id,
        trap_ids=trap_ids,
        retrieved_ids=retrieved,
        true_hit=true_hit,
        trap_hit=trap_hit,
        first_trap_rank=first_trap_rank,
    )


def _aggregate(
    name: str, cases: list[RobustnessCaseResult]
) -> StrategyRobustness:
    n = len(cases)
    if n == 0:
        return StrategyRobustness(name=name)
    true_hits = sum(1 for c in cases if c.true_hit)
    trap_hits = sum(1 for c in cases if c.trap_hit)
    contaminated_ranks = [
        c.first_trap_rank for c in cases if c.first_trap_rank is not None
    ]
    mean_rank = (
        sum(contaminated_ranks) / len(contaminated_ranks)
        if contaminated_ranks
        else None
    )
    return StrategyRobustness(
        name=name,
        true_hit_rate=true_hits / n,
        trap_contamination_rate=trap_hits / n,
        mean_first_trap_rank=mean_rank,
        case_results=cases,
    )


_ROBUSTNESS_CSS = """\
  :root {
    --idea: #2563eb; --base: #64748b; --bg: #f8fafc; --card: #ffffff;
    --ink: #0f172a; --muted: #64748b; --good: #16a34a; --bad: #dc2626;
    --border: #e2e8f0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; color: var(--ink); background: var(--bg);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header { padding: 32px 24px 8px; max-width: 960px; margin: 0 auto; }
  main { padding: 0 24px 48px; max-width: 960px; margin: 0 auto; }
  h1 { margin: 0 0 4px; font-size: 28px; }
  h2 { margin: 0 0 12px; font-size: 18px; }
  .muted { color: var(--muted); }
  .headline {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin: 16px 0 24px;
  }
  .stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; text-align: center;
  }
  .stat .num { display: block; font-size: 26px; font-weight: 700; }
  .stat .label { display: block; font-size: 12px; color: var(--muted); margin-top: 4px; }
  .stat .num.idea { color: var(--idea); }
  .stat .num.base { color: var(--base); }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
  }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  th.idea, td.idea { color: var(--idea); }
  th.base, td.base { color: var(--base); }
  .good { color: var(--good); font-weight: 600; }
  .bad { color: var(--bad); font-weight: 600; }
  footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rank_cell(rank: float | None, cls: str) -> str:
    text = str(rank) if rank is not None else "n/a"
    return f'<td class="{cls}">{text}</td>'


def _render_robustness_html(report: RobustnessReport) -> str:
    ib = report.ideablock
    base = report.baseline
    improvement = report.trap_resistance_improvement
    imp_text = (
        f"{improvement:.2f}x" if improvement != float("inf") else "&infin;"
    )

    header = (
        "<tr>"
        "<th>Metric</th>"
        '<th class="idea">IdeaBlock</th>'
        '<th class="base">Naive chunks</th>'
        "</tr>"
    )
    body = (
        f"<tr><td>True-hit rate (top-{report.k})</td>"
        f'<td class="idea">{_pct(ib.true_hit_rate)}</td>'
        f'<td class="base">{_pct(base.true_hit_rate)}</td></tr>'
        + f"<tr><td>Trap contamination rate</td>"
        f'<td class="idea">{_pct(ib.trap_contamination_rate)}</td>'
        f'<td class="base">{_pct(base.trap_contamination_rate)}</td></tr>'
        + "<tr><td>Mean first-trap rank</td>"
        + _rank_cell(ib.mean_first_trap_rank, "idea")
        + _rank_cell(base.mean_first_trap_rank, "base")
        + "</tr>"
    )

    headline = (
        '<section class="headline">'
        f'<div class="stat"><span class="num idea">{_pct(ib.true_hit_rate)}</span>'
        f'<span class="label">IdeaBlock true-hit</span></div>'
        f'<div class="stat"><span class="num idea">{_pct(ib.trap_contamination_rate)}</span>'
        f'<span class="label">IdeaBlock contamination</span></div>'
        f'<div class="stat"><span class="num base">{_pct(base.trap_contamination_rate)}</span>'
        f'<span class="label">Naive contamination</span></div>'
        + "</section>"
    )

    generated = html.escape(report.generated_at.isoformat())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SparkSage adversarial robustness benchmark</title>
<style>
{_ROBUSTNESS_CSS}</style>
</head>
<body>
<header>
  <h1>Adversarial robustness benchmark</h1>
  <p class="muted">{html.escape(report.summary())}</p>
</header>
<main>
  {headline}
  <section class="card">
    <h2>Strategy comparison</h2>
    <p class="muted">{report.case_count} queries, {report.trap_count} traps
    injected, top-{report.k} window.</p>
    <table><thead>{header}</thead><tbody>{body}</tbody></table>
    <p class="muted">Trap resistance: {html.escape(imp_text)} (IdeaBlock vs naive).</p>
  </section>
</main>
<footer>Generated {generated} &middot; SparkSage eval (robustness)</footer>
</body>
</html>
"""


__all__ = [
    "DEFAULT_ROBUSTNESS_K",
    "DEFAULT_TRAPS_PER_BLOCK",
    "DistractorInjector",
    "RobustnessCaseResult",
    "RobustnessEvaluator",
    "RobustnessReport",
    "StrategyRobustness",
    "TrapRecord",
]
