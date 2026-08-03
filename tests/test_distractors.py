"""Tests for the adversarial distractor injection evaluator.

All tests run fully offline with :class:`FakeEmbeddingClient`: the injector
finds donor candidates via the deterministic hashing embedder, and the
robustness evaluator builds both the IdeaBlock and naive-chunk indexes over the
true + trap corpus the same way the benchmark runner does.
"""

from __future__ import annotations

import json

import pytest

from sparksage import BlockEmbedder, FakeEmbeddingClient, IdeaBlock
from sparksage.eval import (
    DistractorInjector,
    RobustnessEvaluator,
    RobustnessReport,
    StrategyRobustness,
    TrapRecord,
)


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _block(name: str, question: str, answer: str) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


def _corpus() -> list[IdeaBlock]:
    return [
        _block(
            "Deploy",
            "how to deploy sparksage?",
            "install sparksage via pip install sparksage then run the server",
        ),
        _block(
            "Install",
            "how to install sparksage?",
            "use pip install sparksage from the python package index pypi",
        ),
        _block(
            "Config",
            "how to configure the api key?",
            "set the OPENAI_API_KEY environment variable in your shell session",
        ),
        _block(
            "Bake",
            "how to bake a chocolate cake?",
            "mix flour sugar cocoa eggs and butter then bake at 180 degrees",
        ),
    ]


def _embedder(dim: int = 64) -> BlockEmbedder:
    return BlockEmbedder(FakeEmbeddingClient(dimension=dim))


# ---------------------------------------------------------------------------- #
# DistractorInjector
# ---------------------------------------------------------------------------- #
class TestDistractorInjector:
    def test_requires_two_blocks(self):
        injector = DistractorInjector(_embedder())
        assert injector.generate_traps([_block("A", "q?", "a")]) == []

    def test_generates_one_trap_per_block_default(self):
        injector = DistractorInjector(_embedder())
        traps = injector.generate_traps(_corpus())
        assert len(traps) == 4
        assert all(isinstance(t, TrapRecord) for t in traps)

    def test_trap_copies_question_swaps_answer(self):
        injector = DistractorInjector(_embedder())
        blocks = _corpus()
        traps = injector.generate_traps(blocks, traps_per_block=1)
        target = blocks[0]
        trap = next(t for t in traps if t.target_block_id == str(target.id))
        assert trap.trap_block.critical_question == target.critical_question
        assert trap.trap_block.name == target.name
        assert trap.trap_block.trusted_answer != target.trusted_answer

    def test_donor_has_different_answer(self):
        injector = DistractorInjector(_embedder())
        blocks = _corpus()
        traps = injector.generate_traps(blocks)
        for trap in traps:
            target = next(b for b in blocks if str(b.id) == trap.target_block_id)
            donor = next(b for b in blocks if str(b.id) == trap.donor_block_id)
            assert trap.trap_block.trusted_answer == donor.trusted_answer
            assert donor.trusted_answer != target.trusted_answer

    def test_traps_per_block(self):
        injector = DistractorInjector(_embedder())
        traps = injector.generate_traps(_corpus(), traps_per_block=2)
        # each target gets up to 2 traps (limited by corpus size - 1 = 3 donors)
        targets = {t.target_block_id for t in traps}
        assert len(targets) == 4
        # at least some targets got 2 traps
        counts: dict[str, int] = {}
        for t in traps:
            counts[t.target_block_id] = counts.get(t.target_block_id, 0) + 1
        assert max(counts.values()) >= 2

    def test_rejects_bad_traps_per_block(self):
        injector = DistractorInjector(_embedder())
        with pytest.raises(ValueError):
            injector.generate_traps(_corpus(), traps_per_block=0)

    def test_min_similarity_filter(self):
        injector = DistractorInjector(_embedder(), min_similarity=0.99)
        traps = injector.generate_traps(_corpus())
        # such a high threshold filters out all donors
        assert traps == []

    def test_build_adversarial_corpus(self):
        injector = DistractorInjector(_embedder())
        blocks = _corpus()
        corpus, traps = injector.build_adversarial_corpus(blocks)
        assert len(corpus) == len(blocks) + len(traps)
        assert corpus[: len(blocks)] == blocks
        assert all(isinstance(t, TrapRecord) for t in traps)

    def test_skips_identical_answer_donors(self):
        injector = DistractorInjector(_embedder())
        answer = "the exact same answer for both blocks"
        blocks = [
            _block("A", "question alpha?", answer),
            _block("B", "question beta?", answer),
        ]
        traps = injector.generate_traps(blocks)
        assert traps == []

    def test_trap_ids_are_unique(self):
        injector = DistractorInjector(_embedder())
        traps = injector.generate_traps(_corpus(), traps_per_block=2)
        ids = {t.trap_id for t in traps}
        assert len(ids) == len(traps)


# ---------------------------------------------------------------------------- #
# RobustnessEvaluator
# ---------------------------------------------------------------------------- #
class TestRobustnessEvaluator:
    def test_evaluate_returns_report(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        assert isinstance(report, RobustnessReport)
        assert report.case_count == 4
        assert report.trap_count == 4
        assert report.k == 5

    def test_strategies_populated(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=3)
        assert isinstance(report.ideablock, StrategyRobustness)
        assert isinstance(report.baseline, StrategyRobustness)
        assert report.ideablock.name == "IdeaBlock"
        assert report.baseline.name == "Naive chunks"

    def test_ideablock_has_nonzero_true_hit(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        # IdeaBlock should surface the true block for at least some queries
        assert report.ideablock.true_hit_rate > 0.0

    def test_trap_contamination_in_range(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        for s in (report.ideablock, report.baseline):
            assert 0.0 <= s.trap_contamination_rate <= 1.0
            assert 0.0 <= s.true_hit_rate <= 1.0

    def test_case_results_populated(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        assert len(report.ideablock.case_results) == 4
        assert len(report.baseline.case_results) == 4

    def test_empty_corpus_raises(self):
        evaluator = RobustnessEvaluator(_embedder())
        with pytest.raises(ValueError):
            evaluator.evaluate([])

    def test_bad_k_raises(self):
        evaluator = RobustnessEvaluator(_embedder())
        with pytest.raises(ValueError):
            evaluator.evaluate(_corpus(), k=0)

    def test_single_block_returns_empty_report(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate([_block("A", "q?", "a")], k=5)
        assert report.trap_count == 0
        assert report.case_count == 1

    def test_trap_resistance_improvement(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        # The improvement is a valid float or inf
        imp = report.trap_resistance_improvement
        assert imp == 0.0 or imp > 0.0 or imp == float("inf")

    def test_trap_resistance_inf_when_ideablock_clean(self):
        report = RobustnessReport(
            case_count=4,
            trap_count=4,
            k=5,
            ideablock=StrategyRobustness(
                name="IdeaBlock",
                true_hit_rate=1.0,
                trap_contamination_rate=0.0,
            ),
            baseline=StrategyRobustness(
                name="Naive chunks",
                true_hit_rate=0.8,
                trap_contamination_rate=0.5,
            ),
        )
        assert report.trap_resistance_improvement == float("inf")

    def test_trap_resistance_zero_when_both_clean(self):
        report = RobustnessReport(
            case_count=4,
            trap_count=4,
            k=5,
            ideablock=StrategyRobustness(
                name="IdeaBlock", trap_contamination_rate=0.0
            ),
            baseline=StrategyRobustness(
                name="Naive chunks", trap_contamination_rate=0.0
            ),
        )
        assert report.trap_resistance_improvement == 0.0

    def test_config_has_dimension(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        assert report.config["embedding_dimension"] == 64
        assert report.config["k"] == 5

    def test_summary_is_readable(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        s = report.summary()
        assert "robustness" in s.lower()
        assert "contamination" in s.lower()

    def test_to_dict_is_json_safe(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        data = report.to_dict()
        json.dumps(data)

    def test_to_html_is_self_contained(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        html = report.to_html()
        assert html.startswith("<!doctype html>")
        assert "</html>" in html
        assert "IdeaBlock" in html
        assert "src=" not in html

    def test_first_trap_rank_set_when_contaminated(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        contaminated = [
            c for c in report.ideablock.case_results if c.trap_hit
        ]
        for case in contaminated:
            assert case.first_trap_rank is not None
            assert case.first_trap_rank >= 1

    def test_traps_per_block_config(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5, traps_per_block=2)
        assert report.config["traps_per_block"] == 2
        assert report.trap_count >= 4

    def test_trap_similarities_recorded(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        sims = report.config.get("trap_similarities")
        assert isinstance(sims, list)
        assert len(sims) == report.trap_count

    def test_mean_trap_similarity(self):
        evaluator = RobustnessEvaluator(_embedder())
        report = evaluator.evaluate(_corpus(), k=5)
        assert 0.0 <= report.mean_trap_similarity <= 1.0
