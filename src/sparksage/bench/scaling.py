"""Scaling-staircase report: how retrieval quality changes with corpus size.

This is the scale-dependent counterpart to :class:`~sparksage.bench.report.BenchmarkReport`.
Where the single-tier benchmark answers "is IdeaBlock better *here*?", the
scaling staircase answers "does the gap widen or narrow *as the corpus grows*,
and at what size does one strategy overtake the other?" -- the scale-dependent
crossover the paper ("BM25 Wins at Scale") identified as the key finding.

A :class:`ScalingReport` holds one :class:`TierResult` per staircase step
(each step indexes ``blocks[:tier_size]`` against the *same* fixed query set),
plus crossover detection (:meth:`ScalingReport.crossover_tier`) that finds the
first tier where the strategy leader changes for a given metric.

:meth:`ScalingReport.to_html` renders a self-contained page with a per-tier
comparison table and the crossover callout -- the "prove *where* it scales"
artifact.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sparksage.bench.report import StrategyReport


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


@dataclass
class TierResult:
    """One staircase tier: both strategies' metrics at a given corpus size.

    Attributes
    ----------
    tier_index:
        Zero-based position in the staircase.
    block_count:
        Number of blocks indexed at this tier (the corpus slice size).
    ideablock, baseline:
        :class:`~sparksage.bench.report.StrategyReport` for each strategy at
        this tier.
    """

    tier_index: int
    block_count: int
    ideablock: StrategyReport = field(default_factory=StrategyReport)
    baseline: StrategyReport = field(default_factory=StrategyReport)


@dataclass
class ScalingReport:
    """The full nested-staircase scaling outcome.

    Attributes
    ----------
    tiers:
        List of :class:`TierResult`, one per staircase step, ordered by
        increasing corpus size.
    query_count:
        The fixed query-set size (same questions asked at every tier).
    growth_factor:
        The per-tier size multiplier used.
    config:
        Free-form config snapshot (carried from the runner).
    generated_at:
        UTC timestamp the report was produced.
    """

    tiers: list[TierResult] = field(default_factory=list)
    query_count: int = 0
    growth_factor: float = 1.25
    config: dict[str, object] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tier_count(self) -> int:
        """Number of tiers evaluated."""
        return len(self.tiers)

    @property
    def max_block_count(self) -> int:
        """The largest corpus size evaluated (the final tier)."""
        return self.tiers[-1].block_count if self.tiers else 0

    def metric_series(self, metric: str = "hit_at_1") -> list[tuple[int, float, float]]:
        """Return ``(tier_size, ideablock_value, baseline_value)`` per tier.

        ``metric`` is one of ``"hit_at_1"``, ``"hit_at_3"``, ``"hit_at_5"``,
        ``"mrr"``, ``"avg_top_score"``. This is the series the crossover scan
        and the HTML trend chart are built from.
        """
        out: list[tuple[int, float, float]] = []
        for tier in self.tiers:
            iv = _metric_value(tier.ideablock, metric)
            bv = _metric_value(tier.baseline, metric)
            out.append((tier.block_count, iv, bv))
        return out

    def leader(self, tier_index: int, metric: str = "hit_at_1") -> str:
        """Which strategy leads at ``tier_index`` for ``metric``.

        Returns ``"ideablock"``, ``"baseline"``, or ``"tie"`` (within a
        ``1e-9`` tolerance).
        """
        if not self.tiers or tier_index < 0 or tier_index >= len(self.tiers):
            return "tie"
        tier = self.tiers[tier_index]
        iv = _metric_value(tier.ideablock, metric)
        bv = _metric_value(tier.baseline, metric)
        if abs(iv - bv) < 1e-9:
            return "tie"
        return "ideablock" if iv > bv else "baseline"

    def crossover_tier(self, metric: str = "hit_at_1") -> int | None:
        """First tier where the strategy leader changes from the initial leader.

        Returns the tier index, or ``None`` if the same strategy leads at every
        tier (no crossover). This is the scale-dependent crossover the paper
        centers on: the corpus size at which one strategy overtakes the other.
        """
        if not self.tiers:
            return None
        initial = self.leader(0, metric)
        for i in range(1, len(self.tiers)):
            current = self.leader(i, metric)
            if current != initial and current != "tie":
                return i
        return None

    def summary(self, metric: str = "hit_at_1") -> str:
        """One-paragraph summary of the scaling trend + crossover."""
        if not self.tiers:
            return "No tiers evaluated."
        first = self.tiers[0]
        last = self.tiers[-1]
        fi = _metric_value(first.ideablock, metric)
        fb = _metric_value(first.baseline, metric)
        li = _metric_value(last.ideablock, metric)
        lb = _metric_value(last.baseline, metric)
        parts = [
            f"Scaling staircase: {self.tier_count} tiers "
            f"({first.block_count} -> {last.block_count} blocks, "
            f"growth={self.growth_factor:.2f}, {self.query_count} fixed queries)."
        ]
        m_name = _METRIC_LABELS.get(metric, metric)
        parts.append(
            f"{m_name} at first tier: IdeaBlock {_pct(fi)} vs baseline {_pct(fb)}; "
            f"at last tier: IdeaBlock {_pct(li)} vs baseline {_pct(lb)}."
        )
        cross = self.crossover_tier(metric)
        if cross is not None:
            parts.append(
                f"Crossover at tier {cross} "
                f"({self.tiers[cross].block_count} blocks): "
                f"leader changed from '{self.leader(0, metric)}' "
                f"to '{self.leader(cross, metric)}'."
            )
        else:
            parts.append(f"No crossover -- '{self.leader(0, metric)}' leads at every tier.")
        return " ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """Plain (JSON-safe) dict view of the scaling report."""
        return {
            "query_count": self.query_count,
            "growth_factor": self.growth_factor,
            "tier_count": self.tier_count,
            "config": dict(self.config),
            "generated_at": self.generated_at.isoformat(),
            "tiers": [
                {
                    "tier_index": t.tier_index,
                    "block_count": t.block_count,
                    "ideablock": _strategy_dict(t.ideablock),
                    "baseline": _strategy_dict(t.baseline),
                }
                for t in self.tiers
            ],
            "crossover": {
                metric: self.crossover_tier(metric)
                for metric in ("hit_at_1", "mrr")
            },
        }

    def to_html(self) -> str:
        """Render the scaling report as a self-contained HTML document."""
        return _render_scaling_html(self, metric="hit_at_1")


def _metric_value(strategy: StrategyReport, metric: str) -> float:
    r = strategy.retrieval
    if metric.startswith("hit_at_"):
        k = int(metric.split("_")[-1])
        return r.hit_at_k.get(k, 0.0)
    if metric == "mrr":
        return r.mrr
    if metric == "avg_top_score":
        return r.avg_top_score
    raise ValueError(f"unknown metric: {metric}")


_METRIC_LABELS: dict[str, str] = {
    "hit_at_1": "Hit@1",
    "hit_at_3": "Hit@3",
    "hit_at_5": "Hit@5",
    "mrr": "MRR",
    "avg_top_score": "Avg top score",
}


def _strategy_dict(strategy: StrategyReport) -> dict[str, object]:
    r = strategy.retrieval
    return {
        "name": strategy.name,
        "retrieval": {
            "hit_at_k": {str(k): v for k, v in r.hit_at_k.items()},
            "mrr": r.mrr,
            "avg_top_score": r.avg_top_score,
        },
    }


_SCALING_CSS = """\
  :root {
    --idea: #2563eb; --base: #64748b; --bg: #f8fafc; --card: #ffffff;
    --ink: #0f172a; --muted: #64748b; --good: #16a34a; --warn: #ea580c;
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
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
  }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  th.idea, td.idea { color: var(--idea); font-weight: 600; }
  th.base, td.base { color: var(--base); font-weight: 600; }
  .crossover { color: var(--warn); font-weight: 700; }
  .leader { font-weight: 700; }
  footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
"""


def _render_scaling_html(report: ScalingReport, *, metric: str = "hit_at_1") -> str:
    if not report.tiers:
        return (
            "<!doctype html><html><body><p>No tiers evaluated.</p></body></html>"
        )
    m_label = _METRIC_LABELS.get(metric, metric)
    cross = report.crossover_tier(metric)

    rows: list[str] = []
    for tier in report.tiers:
        iv = _metric_value(tier.ideablock, metric)
        bv = _metric_value(tier.baseline, metric)
        leader = report.leader(tier.tier_index, metric)
        is_cross = cross is not None and tier.tier_index == cross
        leader_cell = (
            '<span class="crossover">crossover</span>'
            if is_cross
            else f'<span class="leader">{leader}</span>'
        )
        rows.append(
            "<tr>"
            f"<td>{tier.tier_index}</td>"
            f"<td>{tier.block_count}</td>"
            f'<td class="idea">{_pct(iv)}</td>'
            f'<td class="base">{_pct(bv)}</td>'
            f"<td>{leader_cell}</td>"
            "</tr>"
        )

    header = (
        "<tr>"
        "<th>Tier</th>"
        "<th>Blocks</th>"
        f'<th class="idea">IdeaBlock {html.escape(m_label)}</th>'
        f'<th class="base">Naive {html.escape(m_label)}</th>'
        "<th>Leader</th>"
        "</tr>"
    )
    table = f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"

    config_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in report.config.items()
    )
    config_block = (
        f'<section class="card"><h2>Configuration</h2>'
        f"<table><thead><tr><th>Key</th><th>Value</th></tr></thead>"
        f"<tbody>{config_rows}</tbody></table></section>"
        if config_rows
        else ""
    )

    cross_note = (
        f'<section class="card"><h2>Crossover</h2>'
        f'<p>{html.escape(report.summary(metric))}</p></section>'
    )

    generated = html.escape(report.generated_at.isoformat())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SparkSage scaling benchmark</title>
<style>
{_SCALING_CSS}</style>
</head>
<body>
<header>
  <h1>SparkSage scaling benchmark</h1>
  <p class="muted">{html.escape(report.summary(metric))}</p>
</header>
<main>
  <section class="card">
    <h2>Per-tier {html.escape(m_label)}</h2>
    <p class="muted">Same {report.query_count} queries at every tier;
    only the background corpus grows.</p>
    {table}
  </section>
  {cross_note}
  {config_block}
</main>
<footer>Generated {generated} &middot; SparkSage bench (scaling)</footer>
</body>
</html>
"""


__all__ = ["ScalingReport", "TierResult"]
