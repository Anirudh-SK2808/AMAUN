"""The l4_full_study's judgement rules on hand-built numbers (docs/l4_next_run_prompt.md items 4-8,
15-18, 25; D-018, D-020)."""

import pytest

from amaun.continual import study_criteria as c
from amaun.continual.study_criteria import OptionMeasure as O


def _options(**overrides):
    base = {
        "CREATE": O("CREATE", 0.80, {}, adapters_after=7, training_seconds=100.0, per_example=[0.8] * 20),
        "BACKBONE": O("BACKBONE", 0.60, {}, adapters_after=6, training_seconds=0.0, per_example=[0.6] * 20),
        "REUSE:a": O("REUSE:a", 0.79, {}, adapters_after=6, training_seconds=0.0, per_example=[0.79] * 20),
        "FUSE:a": O("FUSE:a", 0.85, {"t": 0.005}, adapters_after=6, training_seconds=130.0,
                    per_example=[0.85] * 20),
    }
    base.update(overrides)
    return base


def test_relative_scores_and_drops():
    assert c.relative_score(0.4, 0.8) == 0.5
    assert c.relative_score(0.0, 0.0) == 1.0
    assert c.relative_drop(0.8, 0.6) == pytest.approx(0.25)
    assert c.relative_drop(0.8, 0.9) == 0.0
    assert c.is_beneficial(0.98, 0.01) and not c.is_beneficial(0.979, 0.0) and not c.is_beneficial(1.0, 0.011)


def test_acceptability_judges_new_task_and_forgetting_separately():
    options = _options()
    create = options["CREATE"]
    assert c.acceptability(options["REUSE:a"], create)["acceptable"]          # 0.79 >= 0.784
    assert not c.acceptability(options["BACKBONE"], create)["acceptable"]     # new task too low
    forgetting = O("FUSE:b", 0.95, {"t": 0.02}, 6, 130.0, per_example=[0.95] * 20)
    assert not c.acceptability(forgetting, create)["acceptable"]              # a better average cannot hide forgetting


def test_inconclusive_when_the_ci_against_098_create_contains_zero():
    create = O("CREATE", 0.5, {}, 7, 1.0, per_example=[0.0, 1.0] * 10)
    noisy = O("REUSE:a", 0.5, {}, 6, 0.0, per_example=[1.0, 0.0] * 10)
    result = c.acceptability(noisy, create)
    assert result["acceptable"] and result["inconclusive"] is True
    clear = c.acceptability(O("REUSE:a", 0.9, {}, 6, 0.0, per_example=[0.9] * 20),
                            O("CREATE", 0.5, {}, 7, 1.0, per_example=[0.5] * 20))
    assert clear["inconclusive"] is False


def test_oracle_is_the_cheapest_acceptable_option():
    options = _options()
    assert c.choose_oracle(options)["option"] == "REUSE:a"       # 6 adapters, 0 s, acceptable
    del options["REUSE:a"]
    assert c.choose_oracle(options)["option"] == "FUSE:a"        # 6 adapters but training
    del options["FUSE:a"]
    assert c.choose_oracle(options)["option"] == "CREATE"        # CREATE is always acceptable uncapped


def test_cap_makes_create_infeasible_and_replaces_it_with_backbone():
    assert c.executed_option("CREATE", at_cap=True) == "BACKBONE"
    assert c.executed_option("CREATE", at_cap=False) == "CREATE"
    assert c.executed_option("FUSE:a", at_cap=True) == "FUSE:a"
    options = {k: v for k, v in _options().items() if k in ("CREATE", "BACKBONE")}
    oracle = c.choose_oracle(options, at_cap=True)
    assert oracle["option"] == "BACKBONE" and oracle["acceptable"] is False


def test_decision_quality_reports_correctness_shortfall_cost_and_forgetting():
    options = _options()
    oracle = c.choose_oracle(options)["option"]
    create = c.decision_quality("CREATE", options, oracle)
    assert not create["decision_correct"] and create["acceptable"]
    assert create["cost_excess"] == {"adapters": 1, "training_seconds": 100.0}
    reuse = c.decision_quality("REUSE:a", options, oracle)
    assert reuse["decision_correct"] and reuse["quality_shortfall"] == 0.0
    backbone = c.decision_quality("BACKBONE", options, oracle)
    assert not backbone["decision_correct"]
    assert backbone["quality_shortfall"] == pytest.approx(0.98 * 0.80 - 0.60)
    assert c.decision_quality("FUSE:a", options, oracle)["forgetting"] == 0.005


def test_fusion_selection_follows_the_written_rule():
    table = {
        "F-0": [{"relative_new": 2.0, "old_drop": 0.0}] * 6,                 # reference: never eligible
        "F-a": [{"relative_new": 0.9, "old_drop": 0.005}] * 6,
        "F-b": [{"relative_new": 0.99, "old_drop": 0.02}] * 6,               # best new but forgets
        "F-c": [{"relative_new": 0.95, "old_drop": 0.0}] * 4 + [{"relative_new": 1.0, "old_drop": 0.0}] * 2,
    }
    result = c.select_fusion_config(table, ineligible=["F-0"])
    assert result["chosen"] == "F-c" and result["fusion_viable"] is True   # 2 pairs within both limits
    only_forgetful = {"F-b": table["F-b"], "F-x": [{"relative_new": 0.5, "old_drop": 0.015}] * 6}
    fallback = c.select_fusion_config(only_forgetful)
    assert fallback["chosen"] == "F-x" and "lowest mean forgetting" in fallback["rule"]
    assert c.select_fusion_config({"F-a": table["F-a"]})["fusion_viable"] is False


def test_eligibility_marks_a_non_discriminating_task():
    assert c.eligibility(0.21, 0.20)["status"] == "eligible"
    assert c.eligibility(0.20, 0.20)["status"] == "non_discriminating"
    assert c.eligibility(0.1269, 0.1281)["status"] == "non_discriminating"    # run 1's ELI5 ROUGE-L


def test_scores_exactly_on_a_boundary_meet_it_despite_float_error():
    # Each of these is exactly on its limit but computes just past it in floating point.
    on_equivalence = c.relative_score(0.98 * 0.17, 0.17)               # 0.97999...
    on_forgetting = c.relative_drop(1.0, 0.99)                          # 0.01000...01
    assert on_equivalence < 0.98 and on_forgetting > 0.01
    assert c.is_beneficial(on_equivalence, on_forgetting)
    assert c.eligibility(0.42, 0.40)["status"] == "eligible"
    create = O("CREATE", 0.80, {}, adapters_after=7)
    fuse = O("FUSE:a", 0.80, {"t": on_forgetting}, adapters_after=6)
    assert c.acceptability(fuse, create)["acceptable"]
    assert c.choose_oracle({"CREATE": create, "FUSE:a": fuse})["option"] == "FUSE:a"
    assert c.decision_quality("FUSE:a", {"CREATE": create, "FUSE:a": fuse}, "FUSE:a")["acceptable"]
    selection = c.select_fusion_config({"F-a": [{"relative_new": on_equivalence, "old_drop": on_forgetting}] * 2})
    assert selection["fusion_viable"] and "highest mean" in selection["rule"]
    # The slack covers float error only: a real shortfall still fails.
    assert not c.is_beneficial(0.9799, 0.0) and not c.is_beneficial(1.0, 0.0101)
    assert c.eligibility(0.2099, 0.20)["status"] == "non_discriminating"


def test_probe_rule():
    backbone = {"loss": 2.0, "score": 0.5}
    good = {"a": {"loss": 1.5, "score": 0.6}, "b": {"loss": 1.8, "score": 0.4}}
    assert c.probe_rule(backbone, good)["option"] == "REUSE:a"
    worse_loss = {"a": {"loss": 2.5, "score": 0.51}}
    assert c.probe_rule(backbone, worse_loss)["option"] == "BACKBONE"   # 0.5 >= 0.98 * 0.51
    strong = {"a": {"loss": 2.5, "score": 0.9}}
    assert c.probe_rule(backbone, strong)["option"] == "CREATE"


def test_simple_policy_candidate_rules_and_top_k():
    cands = {"a": {"loss": 1.0, "score": 0.3}, "b": {"loss": 1.5, "score": 0.7}, "c": {"loss": 0.9, "score": 0.1}}
    assert c.best_by_probe_score(cands) == "b"
    assert c.best_by_loss_gain(cands, backbone_loss=2.0) == "c"
    assert c.top_k_by_loss_gain(cands, backbone_loss=2.0, k=2) == ["c", "a"]


def test_amaun_order():
    backbone = {"loss": 2.0, "score": 0.2}
    cands = {"a": {"loss": 1.0, "score": 0.6}, "b": {"loss": 1.2, "score": 0.5}}
    common = dict(backbone=backbone, candidates=cands, tau_u=0.5)
    assert c.amaun_order(reuse_choice="b", utilities={"a": 0.9}, **common)["option"] == "REUSE:b"
    assert c.amaun_order(reuse_choice=None, utilities={"a": 0.9, "b": 0.1}, **common)["option"] == "FUSE:a"
    assert c.amaun_order(reuse_choice=None, utilities={"a": 0.4, "b": 0.1}, **common)["option"] == "CREATE"
    strong_backbone = dict(common, backbone={"loss": 2.0, "score": 0.6})
    assert c.amaun_order(reuse_choice=None, utilities={"a": 0.9}, **strong_backbone)["option"] == "BACKBONE"


def test_paired_bootstrap_ci():
    ci = c.paired_bootstrap_ci([1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0], resamples=1000, seed=1)
    assert ci["mean"] == pytest.approx(0.5) and ci["ci_low"] <= 0.5 <= ci["ci_high"] and ci["n"] == 4
    weighted = c.paired_bootstrap_ci([1.0] * 5, [1.0] * 5, weight_b=0.98)
    assert weighted["mean"] == pytest.approx(0.02) and weighted["contains_zero"] is False
    with pytest.raises(ValueError):
        c.paired_bootstrap_ci([1.0], [1.0, 2.0])


# D-027: interval-scale metrics (rescaled BERTScore, mostly negative on ELI5) ------------------


def test_equivalence_bar_is_unchanged_for_non_negative_scores():
    for score in (0.0, 0.05, 0.42, 0.9):
        assert c.equivalence_bar(score) == pytest.approx(0.98 * score)


def test_ratio_bar_would_flip_on_negative_scores_and_gain_retention_does_not():
    create, zs = -0.05, -0.11                                   # CREATE beats zero-shot by 0.06
    assert 0.98 * create > create                               # the old bar sat above CREATE itself
    bar = c.equivalence_bar(create, zero_shot=zs)
    assert bar == pytest.approx(zs + 0.98 * (create - zs)) and zs < bar < create
    backbone = O("BACKBONE", zs, {}, adapters_after=6, per_example=[zs] * 20)
    make = lambda key, score, adapters: O(key, score, {}, adapters_after=adapters, per_example=[score] * 20)
    options = {"CREATE": make("CREATE", create, 7), "BACKBONE": backbone,
               "REUSE:a": make("REUSE:a", -0.0511, 6), "REUSE:b": make("REUSE:b", -0.0525, 6)}
    keeps = c.acceptability(options["REUSE:a"], options["CREATE"], zero_shot=backbone)
    loses = c.acceptability(options["REUSE:b"], options["CREATE"], zero_shot=backbone)
    assert keeps["acceptable"] and not loses["acceptable"]
    assert keeps["gain_retained_vs_create"] == pytest.approx((-0.0511 - zs) / (create - zs))
    assert c.choose_oracle(options, zero_shot=zs)["option"] == "REUSE:a"
    quality = c.decision_quality("REUSE:b", options, "REUSE:a", zero_shot=zs)
    assert quality["quality_shortfall"] == pytest.approx(bar - (-0.0525))


def test_eligibility_and_probes_point_the_right_way_on_negative_scores():
    assert c.eligibility(-0.02, -0.11)["status"] == "eligible"          # a large improvement
    assert c.eligibility(-0.12, -0.11)["status"] == "non_discriminating"
    backbone = {"loss": 2.0, "score": -0.10}
    assert c.probe_rule(backbone, {"a": {"loss": 1.5, "score": -0.0995}})["option"] == "REUSE:a"
    assert c.probe_rule(backbone, {"a": {"loss": 2.5, "score": -0.0999}})["option"] == "BACKBONE"


def test_gain_drop_measures_the_share_of_the_gain_lost():
    assert c.gain_drop(-0.05, -0.08, -0.11) == pytest.approx(0.5)
    assert c.gain_drop(-0.05, -0.04, -0.11) == 0.0
    assert c.relative_drop(0.8, 0.6) == pytest.approx(0.25)            # ratio scale unchanged
