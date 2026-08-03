"""Tests for the measurable benchmark suite.

All tests run fully offline and dependency-free: the splitter and metrics are
pure stdlib, and the runner is exercised through the deterministic
:class:`FakeEmbeddingClient` so the comparison is reproducible without any API
key. They cover the baseline recursive splitter, the retrieval/token metrics,
the HTML report, and the end-to-end runner.
"""

from __future__ import annotations

import pytest

from sparksage import BlockEmbedder, FakeEmbeddingClient, IdeaBlock, SearchHit
from sparksage.bench import (
    BenchmarkReport,
    BenchmarkRunner,
    Chunk,
    RecursiveCharSplitter,
    RetrievalMetrics,
    ScalingReport,
    StrategyReport,
    TierResult,
    approx_tokens,
    evaluate_retrieval,
    token_stats,
)
from sparksage.bench.report import _safe_ratio
from sparksage.bench.runner import _compute_tier_sizes, _default_query_count


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _make_block(
    name: str = "Block",
    question: str = "What is this?",
    answer: str = "A short verified answer about sparksage deployment.",
) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


# ---------------------------------------------------------------------------- #
# RecursiveCharSplitter
# ---------------------------------------------------------------------------- #
class TestRecursiveCharSplitter:
    def test_empty_yields_nothing(self):
        splitter = RecursiveCharSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split_text("") == []
        assert splitter.split("", source="x") == []

    def test_short_text_is_one_chunk(self):
        splitter = RecursiveCharSplitter(chunk_size=100, chunk_overlap=10)
        text = "short text"
        assert splitter.split_text(text) == ["short text"]

    def test_respects_chunk_size(self):
        splitter = RecursiveCharSplitter(chunk_size=20, chunk_overlap=0)
        text = "word " * 50  # ~250 chars
        chunks = splitter.split_text(text)
        assert chunks
        assert all(len(c) <= 20 for c in chunks)
        assert "".join(c.rstrip() for c in chunks)  # non-empty content

    def test_char_fallback_when_no_separator(self):
        splitter = RecursiveCharSplitter(chunk_size=10, chunk_overlap=0)
        text = "abcdefghijklmnopqrstuvwxyz"  # no default separators present
        chunks = splitter.split_text(text)
        assert all(len(c) <= 10 for c in chunks)
        assert len(chunks) >= 2

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            RecursiveCharSplitter(chunk_size=0)
        with pytest.raises(ValueError):
            RecursiveCharSplitter(chunk_size=10, chunk_overlap=-1)
        with pytest.raises(ValueError):
            RecursiveCharSplitter(chunk_size=10, chunk_overlap=10)

    def test_split_attaches_provenance_and_offsets(self):
        splitter = RecursiveCharSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split("some text here", source="doc.md", source_ref_id="b1")
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.source == "doc.md" for c in chunks)
        assert all(c.source_ref_id == "b1" for c in chunks)
        assert chunks[0].start == 0

    def test_split_blocks_tags_each_chunk_with_block_id(self):
        splitter = RecursiveCharSplitter(chunk_size=20, chunk_overlap=0)
        block = _make_block("A", "q a?", "alpha alpha alpha alpha alpha")
        chunks = splitter.split_blocks([block])
        assert chunks
        assert all(c.source_ref_id == str(block.id) for c in chunks)

    def test_split_blocks_skips_empty_text_attr(self):
        splitter = RecursiveCharSplitter(chunk_size=20, chunk_overlap=0)

        class Empty:  # noqa: D106
            embedding_text = ""
            id = "x"

        assert splitter.split_blocks([Empty()]) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- #
# metrics: approx_tokens, token_stats
# ---------------------------------------------------------------------------- #
class TestTokenMetrics:
    def test_approx_tokens_default(self):
        assert approx_tokens("abcdefgh") == 2.0  # 8 chars / 4

    def test_approx_tokens_custom_ratio(self):
        assert approx_tokens("abcdefgh", chars_per_token=2) == 4.0

    def test_approx_tokens_rejects_zero_ratio(self):
        with pytest.raises(ValueError):
            approx_tokens("x", chars_per_token=0)

    def test_token_stats_empty(self):
        stats = token_stats([])
        assert stats.unit_count == 0
        assert stats.total_tokens == 0.0

    def test_token_stats_aggregates(self):
        stats = token_stats(["ab", "abcd"], chars_per_token=2.0)
        assert stats.unit_count == 2
        assert stats.total_chars == 6
        assert stats.avg_chars == 3.0
        assert stats.total_tokens == 3.0
        assert stats.avg_tokens == 1.5


# ---------------------------------------------------------------------------- #
# metrics: evaluate_retrieval
# ---------------------------------------------------------------------------- #
class TestEvaluateRetrieval:
    def _hits(self, *ids: str) -> list[SearchHit]:
        return [SearchHit(block_id=bid, score=1.0 - i * 0.1) for i, bid in enumerate(ids)]

    def test_empty_inputs(self):
        metrics = evaluate_retrieval([], [], k_values=(1, 3))
        assert metrics.query_count == 0
        assert metrics.mrr == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_retrieval([self._hits("a")], [{"a"}, {"b"}])

    def test_empty_k_values_raises(self):
        with pytest.raises(ValueError):
            evaluate_retrieval([], [], k_values=())

    def test_perfect_retrieval(self):
        rankings = [self._hits("a", "b"), self._hits("b", "a")]
        truth = [{"a"}, {"b"}]
        metrics = evaluate_retrieval(rankings, truth, k_values=(1, 2))
        assert metrics.hit_at_1 == 1.0
        assert metrics.mrr == 1.0
        assert metrics.query_count == 2

    def test_hit_at_k_counts_rank(self):
        # truth 'a' is at rank 2 -> hit@2 yes, hit@1 no
        rankings = [self._hits("x", "a")]
        metrics = evaluate_retrieval(rankings, [{"a"}], k_values=(1, 2))
        assert metrics.hit_at_k[1] == 0.0
        assert metrics.hit_at_k[2] == 1.0
        assert metrics.mrr == pytest.approx(0.5)

    def test_missing_truth_scores_zero(self):
        rankings = [self._hits("x", "y")]
        metrics = evaluate_retrieval(rankings, [{"a"}], k_values=(1,))
        assert metrics.hit_at_1 == 0.0
        assert metrics.mrr == 0.0

    def test_any_relevant_in_truth_counts(self):
        # ground truth is a set; either member hit counts
        rankings = [self._hits("a", "z")]
        metrics = evaluate_retrieval(rankings, [{"a", "b"}], k_values=(1,))
        assert metrics.hit_at_1 == 1.0

    def test_avg_top_score(self):
        rankings = [
            [SearchHit(block_id="a", score=0.9)],
            [SearchHit(block_id="b", score=0.5)],
        ]
        metrics = evaluate_retrieval(rankings, [{"a"}, {"b"}], k_values=(1,))
        assert metrics.avg_top_score == pytest.approx(0.7)


# ---------------------------------------------------------------------------- #
# report
# ---------------------------------------------------------------------------- #
class TestReport:
    def _report(self) -> BenchmarkReport:
        ib = StrategyReport(
            name="IdeaBlock",
            retrieval=RetrievalMetrics(
                query_count=10, hit_at_k={1: 0.9, 3: 1.0, 5: 1.0}, mrr=0.95, avg_top_score=0.8
            ),
            tokens=token_stats(["x" * 80] * 10),
        )
        base = StrategyReport(
            name="Naive chunks",
            retrieval=RetrievalMetrics(
                query_count=10, hit_at_k={1: 0.3, 3: 0.6, 5: 0.8}, mrr=0.4, avg_top_score=0.5
            ),
            tokens=token_stats(["x" * 320] * 40),
        )
        return BenchmarkReport(
            ideablock=ib, baseline=base, query_count=10, block_count=10, config={"k": "1,3,5"}
        )

    def test_improvement_factors(self):
        report = self._report()
        assert report.hit_at_1_improvement == pytest.approx(3.0)
        assert report.mrr_improvement > 1.0
        assert report.token_efficiency > 1.0  # baseline chunks are bigger
        assert 0.0 < report.token_reduction < 1.0

    def test_safe_ratio_guards_zero(self):
        assert _safe_ratio(1.0, 0.0) == 0.0
        assert _safe_ratio(4.0, 2.0) == 2.0

    def test_to_dict_is_json_safe(self):
        import json

        data = self._report().to_dict()
        json.dumps(data)  # must not raise

    def test_summary_mentions_numbers(self):
        summary = self._report().summary()
        assert "hit@1" in summary
        assert "blocks" in summary

    def test_to_html_is_self_contained(self):
        html = self._report().to_html()
        assert html.startswith("<!doctype html>")
        assert "</html>" in html
        assert "IdeaBlock" in html
        assert "Naive chunks" in html
        # no external resource references
        assert "src=" not in html
        assert "href=" not in html

    def test_to_html_escapes_config(self):
        report = self._report()
        report.config = {"note": "<script>alert(1)</script>"}
        html = report.to_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------- #
# BenchmarkRunner: end-to-end
# ---------------------------------------------------------------------------- #
class TestBenchmarkRunner:
    def _runner(self, **kwargs: object) -> BenchmarkRunner:
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        return BenchmarkRunner(embedder=embedder, **kwargs)  # type: ignore[arg-type]

    def test_run_requires_blocks(self):
        with pytest.raises(ValueError):
            self._runner().run([])

    def test_run_returns_report(self):
        runner = self._runner(k_values=(1, 3))
        blocks = [
            _make_block("A", "how to deploy?", "deploy sparksage locally"),
            _make_block("B", "how to bake?", "chocolate cake recipe sugar"),
        ]
        report = runner.run(blocks)
        assert isinstance(report, BenchmarkReport)
        assert report.query_count == 2
        assert report.block_count == 2
        assert report.ideablock.retrieval.query_count == 2
        assert report.baseline.retrieval.query_count == 2

    def test_ideablock_retrieves_its_own_question(self):
        # With question-aligned blocks, each block's critical_question should
        # retrieve itself at rank 1 (hit@1 == 1.0) on the IdeaBlock index.
        runner = self._runner(k_values=(1,))
        blocks = [
            _make_block("Deploy", "how to deploy sparksage?", "deploy sparksage locally now"),
            _make_block("Cook", "how to bake chocolate?", "chocolate cake recipe sugar eggs"),
        ]
        report = runner.run(blocks)
        assert report.ideablock.retrieval.hit_at_1 == 1.0
        assert report.ideablock.retrieval.mrr == 1.0

    def test_token_counter_override(self):
        runner = self._runner(token_counter=lambda t: len(t.split()))
        blocks = [_make_block("A"), _make_block("B")]
        report = runner.run(blocks)
        # tokens are now word counts (integers), not the len/4 estimate
        assert report.ideablock.tokens.total_tokens == int(
            report.ideablock.tokens.total_tokens
        )
        assert report.ideablock.tokens.avg_tokens > 0

    def test_k_values_validation(self):
        with pytest.raises(ValueError):
            self._runner(k_values=())

    def test_search_k_validation(self):
        with pytest.raises(ValueError):
            self._runner(search_k=0)

    def test_chars_per_token_validation(self):
        with pytest.raises(ValueError):
            self._runner(chars_per_token=0)

    def test_store_factory_validation(self):
        runner = self._runner(store_factory=lambda d: "not-a-store")
        with pytest.raises(TypeError):
            runner.run([_make_block("A")])

    def test_custom_splitter_used(self):
        splitter = RecursiveCharSplitter(chunk_size=30, chunk_overlap=5)
        runner = self._runner(splitter=splitter)
        report = runner.run([_make_block("A"), _make_block("B")])
        assert report.config["chunk_size"] == 30
        assert report.config["chunk_overlap"] == 5

    def test_report_config_snapshot(self):
        runner = self._runner()
        report = runner.run([_make_block("A")])
        assert report.config["embedding_dimension"] == 64
        assert report.config["splitter"] == "RecursiveCharSplitter"
        assert report.config["k_values"] == [1, 3, 5]


# ---------------------------------------------------------------------------- #
# scaling staircase helpers
# ---------------------------------------------------------------------------- #
class TestTierSizeComputation:
    def test_default_query_count_small_corpus(self):
        assert _default_query_count(1) == 1
        assert _default_query_count(2) == 1
        assert _default_query_count(3) == 1

    def test_default_query_count_third_of_corpus(self):
        assert _default_query_count(9) == 3
        assert _default_query_count(30) == 10

    def test_compute_tier_sizes_grows_geometrically(self):
        sizes = _compute_tier_sizes(10, 100, 2.0, None)
        assert sizes[0] == 10
        assert sizes[-1] == 100  # always ends at max
        # each step roughly doubles (ceil)
        assert sizes[1] == 20
        assert sizes[2] == 40

    def test_compute_tier_sizes_caps_at_max(self):
        sizes = _compute_tier_sizes(10, 25, 2.0, None)
        assert sizes[-1] == 25
        assert all(s <= 25 for s in sizes)

    def test_compute_tier_sizes_deduplicates(self):
        sizes = _compute_tier_sizes(10, 10, 1.25, None)
        assert sizes == [10]

    def test_compute_tier_sizes_respects_max_tiers(self):
        sizes = _compute_tier_sizes(10, 1000, 1.25, max_tiers=3)
        assert len(sizes) == 3

    def test_compute_tier_sizes_growth_125(self):
        sizes = _compute_tier_sizes(8, 40, 1.25, None)
        assert sizes[0] == 8
        assert sizes[-1] == 40
        # 8 -> 10 -> 13 -> 17 -> 22 -> 28 -> 35 -> 40(append full)
        assert len(sizes) >= 5


# ---------------------------------------------------------------------------- #
# scaling report
# ---------------------------------------------------------------------------- #
class TestScalingReport:
    def _tier(
        self,
        idx: int,
        size: int,
        ib_hit1: float,
        base_hit1: float,
    ) -> TierResult:
        return TierResult(
            tier_index=idx,
            block_count=size,
            ideablock=StrategyReport(
                name="IdeaBlock",
                retrieval=RetrievalMetrics(
                    query_count=5, hit_at_k={1: ib_hit1, 3: 1.0, 5: 1.0}, mrr=0.9
                ),
            ),
            baseline=StrategyReport(
                name="Naive chunks",
                retrieval=RetrievalMetrics(
                    query_count=5,
                    hit_at_k={1: base_hit1, 3: 0.8, 5: 0.9},
                    mrr=0.7,
                ),
            ),
        )

    def test_metric_series(self):
        report = ScalingReport(
            tiers=[
                self._tier(0, 10, 0.9, 0.3),
                self._tier(1, 20, 0.8, 0.5),
            ],
            query_count=5,
        )
        series = report.metric_series("hit_at_1")
        assert series == [(10, 0.9, 0.3), (20, 0.8, 0.5)]

    def test_leader_detection(self):
        report = ScalingReport(
            tiers=[self._tier(0, 10, 0.9, 0.3), self._tier(1, 20, 0.5, 0.8)]
        )
        assert report.leader(0, "hit_at_1") == "ideablock"
        assert report.leader(1, "hit_at_1") == "baseline"

    def test_leader_tie(self):
        tier = self._tier(0, 10, 0.5, 0.5)
        report = ScalingReport(tiers=[tier])
        assert report.leader(0, "hit_at_1") == "tie"

    def test_crossover_tier_found(self):
        report = ScalingReport(
            tiers=[
                self._tier(0, 10, 0.9, 0.3),
                self._tier(1, 20, 0.5, 0.8),  # baseline overtakes
            ]
        )
        assert report.crossover_tier("hit_at_1") == 1

    def test_crossover_tier_none_when_no_change(self):
        report = ScalingReport(
            tiers=[
                self._tier(0, 10, 0.9, 0.3),
                self._tier(1, 20, 0.8, 0.5),
            ]
        )
        assert report.crossover_tier("hit_at_1") is None

    def test_crossover_empty_report(self):
        report = ScalingReport(tiers=[])
        assert report.crossover_tier() is None

    def test_summary_mentions_crossover(self):
        report = ScalingReport(
            tiers=[
                self._tier(0, 10, 0.9, 0.3),
                self._tier(1, 20, 0.5, 0.8),
            ],
            query_count=5,
        )
        s = report.summary("hit_at_1")
        assert "Crossover" in s

    def test_to_dict_is_json_safe(self):
        import json

        report = ScalingReport(
            tiers=[self._tier(0, 10, 0.9, 0.3)],
            query_count=5,
            config={"growth_factor": 1.25},
        )
        data = report.to_dict()
        json.dumps(data)

    def test_to_html_is_self_contained(self):
        report = ScalingReport(
            tiers=[self._tier(0, 10, 0.9, 0.3)],
            query_count=5,
        )
        out = report.to_html()
        assert out.startswith("<!doctype html>")
        assert "</html>" in out
        assert "IdeaBlock" in out
        assert "src=" not in out


# ---------------------------------------------------------------------------- #
# BenchmarkRunner.run_scaling: end-to-end
# ---------------------------------------------------------------------------- #
class TestRunScaling:
    def _runner(self, **kwargs: object) -> BenchmarkRunner:
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        return BenchmarkRunner(embedder=embedder, **kwargs)  # type: ignore[arg-type]

    def _blocks(self, n: int) -> list[IdeaBlock]:
        return [
            _make_block(
                f"Block {i}",
                f"what is topic {i} about?",
                f"topic {i} discusses subject number {i} in detail here",
            )
            for i in range(n)
        ]

    def test_requires_blocks(self):
        with pytest.raises(ValueError):
            self._runner().run_scaling([])

    def test_returns_scaling_report(self):
        runner = self._runner(k_values=(1, 3))
        report = runner.run_scaling(self._blocks(20), query_count=5)
        assert isinstance(report, ScalingReport)
        assert report.query_count == 5
        assert report.tier_count >= 2
        assert all(isinstance(t, TierResult) for t in report.tiers)

    def test_last_tier_is_full_corpus(self):
        blocks = self._blocks(30)
        report = self._runner().run_scaling(blocks, query_count=5)
        assert report.tiers[-1].block_count == 30

    def test_first_tier_contains_query_set(self):
        report = self._runner().run_scaling(self._blocks(30), query_count=5)
        assert report.tiers[0].block_count >= 5

    def test_tiers_are_monotonically_increasing(self):
        report = self._runner().run_scaling(self._blocks(50), query_count=5)
        sizes = [t.block_count for t in report.tiers]
        assert sizes == sorted(sizes)
        assert len(sizes) == len(set(sizes))  # no duplicates

    def test_growth_factor_in_config(self):
        report = self._runner().run_scaling(
            self._blocks(20), query_count=5, growth_factor=1.5
        )
        assert report.growth_factor == 1.5
        assert report.config["growth_factor"] == 1.5

    def test_rejects_bad_growth_factor(self):
        with pytest.raises(ValueError):
            self._runner().run_scaling(self._blocks(10), growth_factor=1.0)
        with pytest.raises(ValueError):
            self._runner().run_scaling(self._blocks(10), growth_factor=0.5)

    def test_rejects_bad_query_count(self):
        with pytest.raises(ValueError):
            self._runner().run_scaling(self._blocks(10), query_count=0)
        with pytest.raises(ValueError):
            self._runner().run_scaling(self._blocks(10), query_count=11)

    def test_rejects_min_tier_below_query_count(self):
        with pytest.raises(ValueError):
            self._runner().run_scaling(
                self._blocks(20), query_count=10, min_tier_size=5
            )

    def test_max_tiers_cap(self):
        report = self._runner().run_scaling(
            self._blocks(100), query_count=5, max_tiers=3
        )
        assert report.tier_count <= 3

    def test_fixed_query_set_across_tiers(self):
        report = self._runner().run_scaling(self._blocks(30), query_count=5)
        for tier in report.tiers:
            assert tier.ideablock.retrieval.query_count == 5
            assert tier.baseline.retrieval.query_count == 5

    def test_single_block_corpus(self):
        report = self._runner().run_scaling(self._blocks(1), query_count=1)
        assert report.tier_count == 1
        assert report.tiers[0].block_count == 1
