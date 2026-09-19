"""M12 -- task metrics (criterion H4)."""

import pytest

from amaun.errors import EvaluationError
from amaun.evaluation.task_metrics import (
    compute_all,
    compute_pairwise,
    distinct_n,
    exact_match,
    normalize_answer,
    perplexity_from_loss,
    rouge_l,
    rouge_n,
    token_f1,
)


def test_exact_match_normalisation():
    assert exact_match("The Cat.", "cat") == 1.0
    assert exact_match("a dog", "dog") == 1.0        # articles stripped
    assert exact_match("cat", "dog") == 0.0
    assert normalize_answer("The  QUICK, brown!") == "quick brown"


def test_token_f1_endpoints_and_middle():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert token_f1("abc", "xyz") == pytest.approx(0.0)
    # prediction={cat,sat,on,mat} vs reference={cat,sat}: overlap 2, P=2/4, R=2/2, F1=2/3
    assert token_f1("the cat sat on mat", "a cat sat") == pytest.approx(2 / 3, abs=1e-6)


def test_rouge_endpoints():
    assert rouge_n("a b c", "a b c", 1) == pytest.approx(1.0)
    assert rouge_n("a b c", "x y z", 1) == pytest.approx(0.0)
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert rouge_l("", "something") == pytest.approx(0.0)


def test_rouge_l_uses_longest_common_subsequence():
    # LCS("a b c d", "a x c d") = a,c,d = 3; P=R=3/4; F1=0.75
    assert rouge_l("a b c d", "a x c d") == pytest.approx(0.75, abs=1e-6)


def test_rouge2_needs_consecutive_pairs():
    assert rouge_n("a b c", "a c b", 2) == pytest.approx(0.0)
    assert rouge_n("a b c", "a b c", 2) == pytest.approx(1.0)


def test_distinct_n_measures_diversity():
    assert distinct_n(["a b a b", "a b a b"], 2) < distinct_n(["a b c d", "e f g h"], 2)


def test_perplexity_guards_against_overflow():
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)
    assert perplexity_from_loss(float("inf")) == float("inf")
    assert perplexity_from_loss(1e9) < float("inf")     # clamped, not an OverflowError


def test_unimplemented_metrics_raise_rather_than_substitute():
    """A silently substituted metric would misreport what was measured."""
    for metric in ("bertscore_f1", "llm_judge"):
        with pytest.raises(EvaluationError, match="not implemented"):
            compute_pairwise(metric, ["a"], ["a"])


def test_unknown_metric_raises():
    with pytest.raises(EvaluationError, match="unknown metric"):
        compute_pairwise("bleu_score_v2", ["a"], ["a"])


def test_compute_all_handles_the_declared_metric_set():
    metrics = compute_all(
        ["rouge1", "rougeL", "distinct_2", "perplexity", "bertscore_f1"],
        ["a b c"], ["a b x"], mean_token_loss=1.0,
    )
    assert metrics["rouge1"] > 0
    assert metrics["perplexity"] == pytest.approx(2.718281, abs=1e-4)
    # An unavailable metric is reported as unavailable, not quietly dropped.
    assert "bertscore_f1__unavailable" in metrics


def test_length_mismatch_is_refused():
    with pytest.raises(EvaluationError):
        compute_pairwise("rougeL", ["a", "b"], ["a"])
