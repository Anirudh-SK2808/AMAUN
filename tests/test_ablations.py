"""D-030: ablations of a finished full study, scored against its own measurements.

A dummy full study (fit gate forced open, so its utility models are fitted) is run once; the
ablation pipeline is then checked against it: both identity checks, the feature transforms, A6's
predicted equality, A3's frozen alpha, an unmeasured FUSE choice being measured and scored, and a
resume of Part 2.
"""

import copy
import importlib.util
import json
import shutil
import types
import uuid
from pathlib import Path

import pytest

from amaun.config import load_config, resolve
from amaun.continual import ablations as abl
from amaun.features.semantic import SemanticFeatures
from amaun.manager.task_representation import build_task_representation
from amaun.data.splits import manager_guard

ROOT = Path(__file__).resolve().parents[1]
OPEN_GATE = {"min_pairs": 2, "min_beneficial_fraction": 0.0, "max_beneficial_fraction": 1.0}


def _driver():
    spec = importlib.util.spec_from_file_location("run_ablations", ROOT / "scripts" / "run_ablations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(name, experiment_id, mutate):
    raw = load_config(ROOT / "configs" / "experiments" / name, config_root=ROOT / "configs")
    raw = copy.deepcopy(raw)
    mutate(raw["run_mode_overrides"]["dummy"])
    return resolve(raw, run_mode="dummy", experiment_id=experiment_id)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """A finished dummy full study with fitted utility models (the 'frozen run')."""
    from amaun.continual.full_study import FullStudy
    from amaun.experiment import prepare_data
    from amaun.models.backbone import load_backbone
    from amaun.utils.tracking import ExperimentTracker

    results = tmp_path_factory.mktemp("results")
    experiment_id = f"pytest-abl-frozen-{uuid.uuid4().hex[:8]}"
    cfg = _resolve("l4_full_study.yaml", experiment_id,
                   lambda d: d["full_study"].__setitem__("utility_fit_gate", OPEN_GATE))
    tracker = ExperimentTracker(cfg, base=results).start()
    data = prepare_data(cfg)
    study = FullStudy(cfg=cfg, backbone=load_backbone(cfg), tasks=data.tasks, splits=data.splits, tracker=tracker)
    summary = study.run()
    assert not summary.get("aborted") and not summary["failures"]
    yield {"results": results, "root": tracker.paths.root, "adapters": Path("adapters") / "dummy" / experiment_id,
           "splits": data.splits}
    shutil.rmtree(Path("adapters") / "dummy" / experiment_id, ignore_errors=True)


@pytest.fixture(scope="module")
def ablation_cfg(frozen):
    def mutate(d):
        d["ablations"] = {"frozen_run": str(frozen["root"]), "frozen_adapters": str(frozen["adapters"])}
        d.setdefault("full_study", {})["utility_fit_gate"] = OPEN_GATE
    experiment_id = f"pytest-abl-{uuid.uuid4().hex[:8]}"
    cfg = _resolve("l4_full_study_ablations.yaml", experiment_id, mutate)
    yield cfg
    shutil.rmtree(Path("adapters") / "dummy" / experiment_id, ignore_errors=True)


def test_both_identity_checks_pass_on_a_finished_run(frozen):
    run = abl.FrozenRun(frozen["root"], run_mode="dummy")
    assert run.model("amaun") is not None and run.model("amaun_v0") is not None
    replay = abl.identity_replay(run, equivalence=0.98)
    assert replay["ok"], replay
    scoring = abl.identity_scoring(run, caps=[None, 6], policies=list(run.policy_comparison["no_cap"]),
                                   equivalence=0.98, forgetting=0.01, resamples=1000)
    assert scoring["ok"], scoring["first_differences"]


def test_feature_subsets_keep_contiguous_families(frozen):
    run = abl.FrozenRun(frozen["root"], run_mode="dummy")
    vector = next(iter(run.vectors(run.evaluation_tasks[0]).values()))
    semantic = abl.subset(vector, keep=["semantic"])
    assert semantic.names and all(n.startswith("semantic.") for n in semantic.names)
    assert semantic.family_slices == {"semantic": (0, len(semantic.names))}
    no_hist = abl.subset(vector, drop=["historical"])
    assert not any(n.startswith("historical.") for n in no_hist.names)
    assert len(no_hist.names) == len(vector.names) - sum(n.startswith("historical.") for n in vector.names)


def test_a12_recompute_matches_the_semantic_feature_code(frozen, ablation_cfg):
    task_repr = ablation_cfg.section("task_repr")
    tasks = list(frozen["splits"])[:2]
    embeddings = {t: abl.token_stats_embedding(t, frozen["splits"], task_repr) for t in tasks}
    reference = {t: build_task_representation(t, manager_guard(frozen["splits"][t]),
                                              config={**task_repr, "encoder": "token_stats"}).embedding for t in tasks}
    assert embeddings == reference
    ctx = types.SimpleNamespace(task_embedding=embeddings[tasks[0]], candidate_task_embedding=embeddings[tasks[1]])
    expected = SemanticFeatures({"include_cosine_as_feature": True}).compute(ctx)
    run = abl.FrozenRun(frozen["root"], run_mode="dummy")
    vector = next(iter(run.vectors(run.evaluation_tasks[0]).values()))
    swapped = abl.with_task_representation(vector, embeddings[tasks[0]], embeddings[tasks[1]]).as_dict()
    for name in abl.TASK_REPRESENTATION_FEATURES:
        assert swapped[name] == pytest.approx(expected[name.split(".", 1)[1]], rel=1e-6)
    untouched = [n for n in vector.names if n not in abl.TASK_REPRESENTATION_FEATURES]
    assert all(swapped[n] == vector.as_dict()[n] for n in untouched)


def test_part1_scores_every_variant_and_a6_reproduces_the_run(frozen, ablation_cfg):
    from amaun.utils.tracking import ExperimentTracker

    tracker = ExperimentTracker(ablation_cfg, base=frozen["results"]).start()
    summary = _driver().part1(ablation_cfg, tracker, splits=frozen["splits"])
    assert summary["identity"] == {"replay": True, "scoring": True}
    out = json.loads((tracker.paths.root / "ablations" / "feature_variants.json").read_text())
    assert set(out["variants"]) == {"A1", "A6", "A7", "A12"}
    run = abl.FrozenRun(frozen["root"], run_mode="dummy")
    for task, ev in run.evaluations.items():               # A6: historical features are constant -> no change
        for policy in abl.UTILITY_POLICIES:
            assert out["variants"]["A6"]["replayed"][task]["decisions"][policy]["option"] == ev["decisions"][policy]["option"]
    assert out["variants"]["A1"]["replayed"][run.evaluation_tasks[0]]["n_features"] < len(
        next(iter(run.vectors(run.evaluation_tasks[0]).values())).names)


def test_part2_measures_fusion_variants_and_unmeasured_fuse_choices_and_resumes(frozen, ablation_cfg):
    from amaun.utils.tracking import ExperimentTracker

    tracker = ExperimentTracker(ablation_cfg, base=frozen["results"]).start()
    driver = _driver()
    driver.part1(ablation_cfg, tracker, splits=frozen["splits"])
    run = abl.FrozenRun(frozen["root"], run_mode="dummy")
    # Inject one FUSE option the run did not measure.
    task = run.evaluation_tasks[0]
    measured = {k.partition(":")[2] for k in run.evaluations[task]["options"] if k.startswith("FUSE:")}
    candidate = sorted(set(run.seed_task) - measured)[0]
    (tracker.paths.root / "ablations" / "unmeasured_fuse.json").write_text(json.dumps([[task, candidate]]))

    result = driver.part2(ablation_cfg, tracker, types.SimpleNamespace(resume=False))
    assert result["extra_options"] == {task: [f"FUSE:{candidate}"]}
    fusion = json.loads((tracker.paths.root / "ablations" / "fusion_variants.json").read_text())
    assert set(fusion["variants"]) == {"A3", "A8"} and all(len(v["rows"]) == 6 for v in fusion["variants"].values())
    for row in fusion["variants"]["A3"]["rows"]:             # A3: alpha never leaves its initial value
        assert not row["fusion"]["alpha_moved_from_init"]
    assert fusion["variants"]["A8"]["config"]["fusion"]["coefficient"]["granularity"] == "global"
    extra = json.loads((tracker.paths.root / "ablations" / "extra_options.json").read_text())
    assert extra[task][f"FUSE:{candidate}"]["new_task_score"] is not None

    state_path = tracker.paths.root / "decision_study" / "checkpoint" / "state.json"
    units_before = set(json.loads(state_path.read_text())["units"])
    again = driver.part2(ablation_cfg, tracker, types.SimpleNamespace(resume=True))     # every unit cached
    assert set(json.loads(state_path.read_text())["units"]) == units_before
    assert again["extra_options"] == result["extra_options"]           # still scored after a resume
    leftovers = json.loads((tracker.paths.root / "ablations" / "unmeasured_after_part2.json").read_text())
    assert [task, candidate] not in leftovers
    state = json.loads(state_path.read_text())
    assert state["seeded_from"]["run"] == str(frozen["root"]) and len(state["resumes"]) == 1
