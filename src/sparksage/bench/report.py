"""Benchmark report model + zero-dependency HTML renderer.

A :class:`BenchmarkReport` holds the side-by-side retrieval-quality and
token-efficiency numbers for the two indexing strategies (IdeaBlock vs naive
chunking) over the same query set, plus the relative improvement factors that
make the ROI case. :meth:`BenchmarkReport.to_html` renders a self-contained
HTML page (no external CSS/JS, no template engine) so the ROI can be shared as a
single file -- the same "prove it on your own data" narrative that makes
benchmarks worth running.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sparksage.bench.metrics import RetrievalMetrics, TokenStats


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio ``numerator / denominator`` guarded against zero (returns ``0.0``)."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _xfactor(value: float) -> str:
    return f"{value:.2f}x"


@dataclass
class StrategyReport:
    """One strategy's slice of a :class:`BenchmarkReport`.

    Attributes
    ----------
    name:
        Human-readable strategy name ("IdeaBlock" / "Naive chunks").
    retrieval:
        :class:`~sparksage.bench.metrics.RetrievalMetrics` for this strategy.
    tokens:
        :class:`~sparksage.bench.metrics.TokenStats` for this strategy.
    """

    name: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    tokens: TokenStats = field(default_factory=TokenStats)


@dataclass
class BenchmarkReport:
    """Full side-by-side outcome of a benchmark run.

    Attributes
    ----------
    ideablock, baseline:
        :class:`StrategyReport` for each strategy.
    query_count:
        Number of queries evaluated (both strategies ran the same set).
    block_count:
        Number of IdeaBlocks in the corpus.
    config:
        Free-form config snapshot (chunk_size, embedder dimension, k_values, ...).
    generated_at:
        UTC timestamp the report was produced (for the HTML footer).
    """

    ideablock: StrategyReport
    baseline: StrategyReport
    query_count: int = 0
    block_count: int = 0
    config: dict[str, object] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def hit_at_1_improvement(self) -> float:
        """How many times better IdeaBlock hit@1 is vs the baseline (``0`` if n/a)."""
        return _safe_ratio(
            self.ideablock.retrieval.hit_at_1, self.baseline.retrieval.hit_at_1
        )

    @property
    def mrr_improvement(self) -> float:
        """MRR improvement factor (IdeaBlock over baseline)."""
        return _safe_ratio(self.ideablock.retrieval.mrr, self.baseline.retrieval.mrr)

    @property
    def avg_top_score_improvement(self) -> float:
        """Mean-top-score improvement factor (tighter query clusters)."""
        return _safe_ratio(
            self.ideablock.retrieval.avg_top_score,
            self.baseline.retrieval.avg_top_score,
        )

    @property
    def token_efficiency(self) -> float:
        """How many times fewer tokens per unit IdeaBlocks use vs the baseline.

        ``> 1.0`` means IdeaBlocks are more token-efficient (smaller units, so a
        cheaper index and shorter LLM context at retrieval time).
        """
        return _safe_ratio(self.baseline.tokens.avg_tokens, self.ideablock.tokens.avg_tokens)

    @property
    def token_reduction(self) -> float:
        """Fraction of total baseline tokens saved by the IdeaBlock index."""
        base = self.baseline.tokens.total_tokens
        idea = self.ideablock.tokens.total_tokens
        if base == 0:
            return 0.0
        return max(0.0, (base - idea) / base)

    def to_dict(self) -> dict[str, object]:
        """Plain (JSON-safe) dict view of the report."""
        return {
            "query_count": self.query_count,
            "block_count": self.block_count,
            "config": dict(self.config),
            "generated_at": self.generated_at.isoformat(),
            "strategies": {
                "ideablock": _strategy_dict(self.ideablock),
                "baseline": _strategy_dict(self.baseline),
            },
            "improvements": {
                "hit_at_1": self.hit_at_1_improvement,
                "mrr": self.mrr_improvement,
                "avg_top_score": self.avg_top_score_improvement,
                "token_efficiency": self.token_efficiency,
                "token_reduction": self.token_reduction,
            },
        }

    def summary(self) -> str:
        """One-paragraph human-readable summary of the headline numbers."""
        return (
            f"IdeaBlock vs naive chunking over {self.query_count} queries "
            f"({self.block_count} blocks): "
            f"hit@1 {_pct(self.ideablock.retrieval.hit_at_1)} vs "
            f"{_pct(self.baseline.retrieval.hit_at_1)} "
            f"({_xfactor(self.hit_at_1_improvement)}), "
            f"MRR {self.ideablock.retrieval.mrr:.3f} vs "
            f"{self.baseline.retrieval.mrr:.3f} "
            f"({_xfactor(self.mrr_improvement)}), "
            f"avg tokens/unit {self.ideablock.tokens.avg_tokens:.0f} vs "
            f"{self.baseline.tokens.avg_tokens:.0f} "
            f"({_xfactor(self.token_efficiency)} more efficient)."
        )

    def to_html(self) -> str:
        """Render the report as a self-contained HTML document (no externals)."""
        return _render_html(self)


def _strategy_dict(strategy: StrategyReport) -> dict[str, object]:
    r = strategy.retrieval
    t = strategy.tokens
    return {
        "name": strategy.name,
        "retrieval": {
            "query_count": r.query_count,
            "hit_at_k": {str(k): v for k, v in r.hit_at_k.items()},
            "mrr": r.mrr,
            "avg_top_score": r.avg_top_score,
        },
        "tokens": {
            "unit_count": t.unit_count,
            "total_chars": t.total_chars,
            "avg_chars": t.avg_chars,
            "total_tokens": t.total_tokens,
            "avg_tokens": t.avg_tokens,
        },
    }


_COMPARISON_HEADER = (
    "<tr>"
    "<th>Metric</th>"
    '<th class="idea">IdeaBlock</th>'
    '<th class="base">Naive chunks</th>'
    "<th>Improvement</th>"
    "</tr>"
)


_CSS = """\
  :root {
    --idea: #2563eb; --base: #64748b; --bg: #f8fafc; --card: #ffffff;
    --ink: #0f172a; --muted: #64748b; --good: #16a34a; --border: #e2e8f0;
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
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 16px 0 24px;
  }
  .stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; text-align: center;
  }
  .stat .num {
    display: block; font-size: 26px; font-weight: 700; color: var(--idea);
  }
  .stat .label {
    display: block; font-size: 12px; color: var(--muted); margin-top: 4px;
  }
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
  footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
"""


def _stat_card(value: str, label: str) -> str:
    return (
        '<div class="stat">'
        f'<span class="num">{html.escape(value)}</span>'
        f'<span class="label">{html.escape(label)}</span>'
        '</div>'
    )


def _table(header: str, body: str) -> str:
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


def _render_html(report: BenchmarkReport) -> str:
    ib = report.ideablock
    base = report.baseline
    k_values = ib.retrieval.k_values

    config_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in report.config.items()
    )
    hit_rows = "".join(
        _metric_row(
            f"Hit@{k}",
            ib.retrieval.hit_at_k.get(k, 0.0),
            base.retrieval.hit_at_k.get(k, 0.0),
            fmt=_pct,
        )
        for k in k_values
    )
    mrr_row = _metric_row(
        "MRR", ib.retrieval.mrr, base.retrieval.mrr, higher_better=True
    )
    score_row = _metric_row(
        "Avg top-1 score",
        ib.retrieval.avg_top_score,
        base.retrieval.avg_top_score,
        higher_better=True,
    )
    retrieval_table = _table(
        _COMPARISON_HEADER, hit_rows + mrr_row + score_row
    )
    retrieval_intro = (
        f"Same {report.query_count} queries, same embedder, same ground truth. "
        "Only the chunking strategy differs."
    )
    retrieval_block = (
        '<section class="card"><h2>Retrieval quality</h2>'
        f'<p class="muted">{html.escape(retrieval_intro)}</p>'
        f"{retrieval_table}</section>"
    )

    token_body = (
        _metric_row(
            "Indexed units",
            float(ib.tokens.unit_count),
            float(base.tokens.unit_count),
            higher_better=False,
        )
        + _metric_row(
            "Avg chars / unit",
            ib.tokens.avg_chars,
            base.tokens.avg_chars,
            higher_better=False,
        )
        + _metric_row(
            "Avg tokens / unit",
            ib.tokens.avg_tokens,
            base.tokens.avg_tokens,
            higher_better=False,
        )
        + _metric_row(
            "Total tokens",
            ib.tokens.total_tokens,
            base.tokens.total_tokens,
            higher_better=False,
        )
    )
    token_note = (
        f"Tokens estimated at {ib.tokens.chars_per_token:.0f} chars/token. "
        "Override BenchmarkRunner(token_counter=...) for a real tokenizer."
    )
    token_block = (
        '<section class="card"><h2>Token efficiency</h2>'
        f"{_table(_COMPARISON_HEADER, token_body)}"
        f'<p class="muted">{html.escape(token_note)}</p></section>'
    )

    headline_block = (
        '<section class="headline">'
        + _stat_card(_xfactor(report.hit_at_1_improvement), "hit@1 improvement")
        + _stat_card(_xfactor(report.mrr_improvement), "MRR improvement")
        + _stat_card(_xfactor(report.token_efficiency), "token efficiency")
        + _stat_card(_pct(report.token_reduction), "total tokens saved")
        + "</section>"
    )

    if config_rows:
        config_block = (
            '<section class="card"><h2>Configuration</h2>'
            + _table(
                "<tr><th>Key</th><th>Value</th></tr>", config_rows
            )
            + "</section>"
        )
    else:
        config_block = ""

    generated = html.escape(report.generated_at.isoformat())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SparkSage benchmark: IdeaBlock vs naive chunking</title>
<style>
{_CSS}</style>
</head>
<body>
<header>
  <h1>SparkSage retrieval benchmark</h1>
  <p class="muted">{html.escape(report.summary())}</p>
</header>
<main>
  {headline_block}
  {retrieval_block}
  {token_block}
  {config_block}
</main>
<footer>Generated {generated} &middot; SparkSage bench</footer>
</body>
</html>
"""


def _metric_row(
    label: str,
    idea: float,
    base: float,
    *,
    higher_better: bool = True,
    fmt: object | None = None,
) -> str:
    """Render one comparison table row with a relative-improvement cell."""
    format_value = fmt if callable(fmt) else (lambda v: f"{v:.4f}")
    factor = _safe_ratio(idea, base)
    if higher_better:
        better = factor >= 1.0
        improvement_text = f"{_xfactor(factor)} better" if factor != 0 else "n/a"
    else:
        better = idea <= base if base else False
        reduction = (1.0 - _safe_ratio(idea, base)) * 100 if base else 0.0
        improvement_text = f"{reduction:.1f}% smaller" if base else "n/a"
    cls = "good" if better else ""
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f'<td class="idea">{html.escape(format_value(idea))}</td>'
        f'<td class="base">{html.escape(format_value(base))}</td>'
        f'<td class="{cls}">{html.escape(improvement_text)}</td>'
        "</tr>"
    )


__all__ = ["BenchmarkReport", "StrategyReport"]
