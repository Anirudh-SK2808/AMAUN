"""BERTScore (D-021) on the offline random encoder: the bert_score algorithm, rescaling, and wiring."""

import pytest

from amaun.evaluation.bertscore import BERTScoreConfig, BERTScorer
from amaun.evaluation.task_metrics import compute_all, compute_with_examples


def _scorer(rescale: bool, baseline: float | None = 0.8) -> BERTScorer:
    config = BERTScoreConfig(model_id=None, revision=None, num_layers=2, rescale_with_baseline=rescale,
                             baseline_f=baseline, batch_size=2, max_length=64)
    return BERTScorer(config, device="cpu")


def test_identical_text_scores_one_and_empty_scores_zero_before_rescaling():
    scorer = _scorer(rescale=False)
    same, empty = scorer.score(["the cat sat", ""], ["the cat sat", "anything"])
    assert same == pytest.approx(1.0, abs=1e-5)
    assert empty == 0.0
    other = scorer.score(["dogs bark loudly"], ["the cat sat"])[0]
    assert -1.0 <= other < 1.0


def test_rescaling_uses_the_baseline():
    raw = _scorer(rescale=False).score(["a b c"], ["a b d"])[0]
    rescaled = _scorer(rescale=True, baseline=0.8).score(["a b c"], ["a b d"])[0]
    assert rescaled == pytest.approx((raw - 0.8) / (1 - 0.8), abs=1e-5)


def test_rescaling_requires_a_baseline():
    with pytest.raises(Exception):
        BERTScoreConfig.from_config({"num_layers": 2, "rescale_with_baseline": True, "batch_size": 1,
                                     "max_length": 8})


def test_metric_registry_uses_a_scorer_when_one_is_configured():
    scorer = _scorer(rescale=False)
    metrics, per_example = compute_with_examples(["bertscore_f1", "rougeL"], ["x y", "z"], ["x y", "q"],
                                                 scorers={"bertscore_f1": scorer})
    assert len(per_example["bertscore_f1"]) == 2 and per_example["bertscore_f1"][0] == pytest.approx(1.0, abs=1e-5)
    assert metrics["bertscore_f1"] == pytest.approx(sum(per_example["bertscore_f1"]) / 2)
    # Without a scorer the metric is reported unavailable, never substituted.
    assert "bertscore_f1__unavailable" in compute_all(["bertscore_f1"], ["a"], ["a"])


def test_encoder_is_held_on_cpu_between_scoring_calls():
    """The encoder moves to the scoring device only inside ``score`` (L4 memory: D-026 note)."""
    from amaun.evaluation.bertscore import BERTScoreConfig, BERTScorer

    scorer = BERTScorer(BERTScoreConfig.from_config({
        "model_id": None, "revision": None, "num_layers": 2, "rescale_with_baseline": False,
        "baseline_f": 0.0, "batch_size": 4, "max_length": 32, "cache_dir": None,
        "local_files_only": False}), device="cpu").load()
    moves = []
    real_to = scorer._model.to
    scorer._model.to = lambda device, *a, **k: (moves.append(str(device)), real_to(device, *a, **k))[1]
    scorer.score(["a cat sat"], ["a cat sat down"])
    assert moves == ["cpu", "cpu"]                  # to the scoring device, then back to cpu
    assert next(scorer._model.parameters()).device.type == "cpu"
