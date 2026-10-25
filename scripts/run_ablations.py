#!/usr/bin/env python
"""Ablations of the finished l4_full_study (D-030), scored against that run's own measurements.

    .venv/bin/python scripts/run_ablations.py --config configs/experiments/l4_full_study_ablations.yaml \
        --run-mode local_smoke --part 1            # A1 / A6 / A7 / A12, offline (no GPU)
    ... --part 2 [--resume]                         # A3 / A8 + unmeasured FUSE choices (GPU), then re-score

Part 1 gates on two identity checks (replay with the run's frozen models reproduces its logged
decisions; re-scoring its logged decisions reproduces its stored tables) and stops if either fails.
Part 2 is checkpointed per unit and resumable.  The frozen run is only read.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.continual import study_criteria as criteria  # noqa: E402
from amaun.continual.ablations import (  # noqa: E402
    FeatureVariant,
    FrozenModel,
    FrozenRun,
    identity_replay,
    identity_scoring,
    replay_decisions,
    score_with_decisions,
    supervision_from_frozen,
    token_stats_embedding,
    unmeasured_fuse,
)
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args  # noqa: E402
from amaun.manager.utility_cv import FitGate, fit_revised, revised_specs  # noqa: E402
from amaun.utils.tracking import ExperimentTracker  # noqa: E402

UTILITY_POLICIES = ("amaun", "amaun_train_first")


def rules(cfg) -> dict[str, Any]:
    full = cfg.section("full_study")
    c = full["criteria"]
    return {"caps": [None if x is None else int(x) for x in full["caps"]], "policies": list(full["policies"]),
            "equivalence": float(c["equivalence"]), "forgetting": float(c["forgetting"]),
            "resamples": int(c["bootstrap_resamples"])}


def part1(cfg, tracker, *, splits: Mapping[str, Any] | None,
          extra_options: Mapping[str, Mapping[str, Any]] | None = None,
          unmeasured_file: str = "unmeasured_fuse.json") -> dict[str, Any]:
    """Identity checks, then every feature variant.  ``unmeasured_file`` is Part 2's input list
    on the first pass; the re-score after Part 2 writes its leftovers elsewhere, so a resumed
    Part 2 still sees everything it has to measure."""
    ab, r = cfg.section("ablations"), rules(cfg)
    run = FrozenRun(ab["frozen_run"], run_mode=cfg.run_mode.value)
    identity = {"replay": identity_replay(run, equivalence=r["equivalence"]),
                "scoring": identity_scoring(run, caps=r["caps"], policies=r["policies"], equivalence=r["equivalence"],
                                            forgetting=r["forgetting"], resamples=r["resamples"])}
    tracker.write_json("ablations/identity_checks.json", identity)
    if not (identity["replay"]["ok"] and identity["scoring"]["ok"]):
        raise SystemExit("identity check failed (ablations/identity_checks.json); nothing was scored")

    variants = [FeatureVariant.from_config(n, s) for n, s in (ab.get("feature_variants") or {}).items()]
    embeddings = None
    if any(v.task_representation for v in variants):
        if splits is None:
            raise SystemExit("A12 needs the data splits (token_stats embeddings)")
        tasks = sorted(set(run.seed_task.values()) | set(run.evaluation_tasks))
        embeddings = {t: token_stats_embedding(t, splits, cfg.section("task_repr")) for t in tasks}
    full = cfg.section("full_study")
    gate, specs = FitGate.from_config(full["utility_fit_gate"]), revised_specs(full["utility_models"])
    reference = score_with_decisions(run, {t: e["decisions"] for t, e in run.evaluations.items()},
                                     extra_options=extra_options, **{k: r[k] for k in r if k != "policies"},
                                     policies=r["policies"])
    results, unmeasured = {}, set()
    for variant in variants:
        transform = variant.transform(embeddings)
        revised = fit_revised(cfg, supervision_from_frozen(run, transform), gate=gate, specs=specs)
        decisions_by_task, replayed = {}, {}
        for task_id, evaluation in run.evaluations.items():
            vectors = {c: transform(v, task_id, run.seed_task[c]) for c, v in run.vectors(task_id).items()}
            if revised.fitted:
                frozen = FrozenModel(revised.model, revised.normalizer, revised.thresholds)
                decided, utilities = replay_decisions(evaluation, vectors, frozen, run_mode=run.run_mode,
                                                      equivalence=r["equivalence"], policies=UTILITY_POLICIES)
            else:   # the study's automatic fallback
                rule = criteria.probe_rule(evaluation["probes"]["backbone"], evaluation["probes"]["candidates"],
                                           equivalence=r["equivalence"])
                decided = {p: {**rule, "fallback": "probe_rule (utility network not fitted)"} for p in UTILITY_POLICIES}
                utilities = None
            decisions_by_task[task_id] = {**evaluation["decisions"], **decided}
            replayed[task_id] = {"decisions": decided, "utilities": utilities,
                                 "n_features": len(next(iter(vectors.values())).names)}
        scored = score_with_decisions(run, decisions_by_task, extra_options=extra_options,
                                      **{k: r[k] for k in r if k != "policies"}, policies=r["policies"])
        missing = unmeasured_fuse(scored["per_task"], UTILITY_POLICIES)
        unmeasured |= set(missing)
        results[variant.name] = {
            "variant": variant.__dict__, "utility_models": revised.as_dict(), "replayed": replayed,
            "pooled": {cap: {p: scored["pooled"][cap][p] for p in UTILITY_POLICIES}
                       for cap in scored["pooled"] if cap.startswith(("no_cap", "cap_"))},
            "per_task": {t: {cap: {p: o["policies"].get(p) for p in UTILITY_POLICIES} for cap, o in outcomes.items()}
                         for t, outcomes in scored["per_task"].items()},
            "unmeasured_fuse": missing}
    payload = {"frozen_run": str(run.root), "reference": {cap: {p: reference["pooled"][cap][p] for p in r["policies"]}
                                                          for cap in reference["pooled"] if cap.startswith(("no_cap", "cap_"))},
               "eligible_tasks": run.policy_comparison["eligible_tasks"], "variants": results,
               "embeddings": embeddings, "extra_options_used": sorted((extra_options or {}).keys())}
    tracker.write_json("ablations/feature_variants.json", payload)
    tracker.write_json(f"ablations/{unmeasured_file}", sorted(unmeasured))
    return {"identity": {k: v["ok"] for k, v in identity.items()}, "unmeasured_fuse": sorted(unmeasured),
            "variants": {n: {"selected": v["utility_models"].get("selected"),
                             "no_cap_correct": v["pooled"]["no_cap"]["amaun"]["decision_correct"]}
                         for n, v in results.items()}}


def part2(cfg, tracker, args) -> dict[str, Any]:
    from amaun.adapters.repository import AdapterRepository
    from amaun.config import _deep_merge
    from amaun.continual.full_study import FullStudy, FusionVariant
    from amaun.fusion.layerwise import FusionConfig
    from amaun.models.backbone import load_backbone
    from amaun.training.kd import KDConfig
    from amaun.utils.seed import configure_determinism

    ab = cfg.section("ablations")
    configure_determinism(bool(cfg.get("experiment.deterministic", False)))
    data = prepare_data(cfg)
    backbone = load_backbone(cfg)
    study = FullStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker,
                      resume=args.resume)
    run = FrozenRun(ab["frozen_run"], run_mode=cfg.run_mode.value)
    if not study.ckpt.resumed:
        # Seed with the frozen run's measurement caches and trained-adapter records: re-measuring
        # them would only reproduce them (and the trained adapters are read from where they are).
        frozen_state = json.loads((run.d / "checkpoint" / "state.json").read_text())
        study.ckpt.state["caches"] = copy.deepcopy(frozen_state["caches"])
        for key, unit in frozen_state["units"].items():
            if key.startswith("P5:train:"):
                study.ckpt.state["units"][key] = unit
        study.ckpt.state["seeded_from"] = {"run": str(run.root), "config_hash": frozen_state["config_hash"]}
        study.ckpt.save()
    study.repository = AdapterRepository.load(ab["frozen_adapters"], backbone_fingerprint=backbone.fingerprint,
                                              journal_path=tracker.paths.repository_events)
    study.runner.repository = study.repository
    study.selected = study.full.variant(run.fusion_selection["chosen"])

    base = study.full.variant(ab["fusion_base"])
    fusion_rows = {}
    for spec in ab.get("fusion_variants") or []:
        fusion_raw = _deep_merge(base.raw["fusion"], spec.get("fusion") or {})
        kd_raw = _deep_merge(base.raw["kd"], spec.get("kd") or {})
        variant = FusionVariant(spec["name"], False,
                                FusionConfig.from_config(fusion_raw, lora_learning_rate=study.runner.train_cfg.learning_rate),
                                KDConfig.from_config(kd_raw), {"fusion": fusion_raw, "kd": kd_raw})
        rows = []
        for new_id, old_id in itertools.permutations(study.full.fusion_pair_adapters, 2):
            rows.append(study._unit(f"ABL:{variant.name}:{new_id}->{old_id}",
                                    lambda v=variant, a=new_id, b=old_id: study._pair(v, a, b, f"abl-{v.name}",
                                                                                      supervision=False)))
        fusion_rows[variant.name] = {"config": variant.raw, "rows": rows}
    table = {name: [{"relative_new": x["relative_new"], "old_drop": x["old_drop"]} for x in v["rows"]]
             for name, v in fusion_rows.items()}
    for name in (ab["fusion_base"], "F-e"):
        table[name] = [{"relative_new": x["relative_new"], "old_drop": x["old_drop"]}
                       for x in run.fusion_selection["rows"][name]]
    summary = criteria.select_fusion_config(table, forgetting=study.full.forgetting,
                                            equivalence=study.full.equivalence)["summary"]
    tracker.write_json("ablations/fusion_variants.json", {"base": ab["fusion_base"], "variants": fusion_rows,
                                                          "summary_with_frozen_rows": summary,
                                                          "note": "summary only; the run's chosen config is not re-selected"})

    # Part 1's list plus every option already measured here (a resumed Part 2 keeps them all).
    listed = {tuple(x) for x in json.loads((tracker.paths.root / "ablations" / "unmeasured_fuse.json").read_text())}
    done = {(k.split(":")[2], k.split(":", 4)[4]) for k in study.ckpt.state["units"] if k.startswith("ABL:option:")}
    unmeasured = sorted(listed | done)
    extra: dict[str, dict[str, Any]] = {}
    for task_id, candidate in unmeasured:
        temp = study._temp(task_id)
        key = f"FUSE:{candidate}"
        extra.setdefault(task_id, {})[key] = study._unit(
            f"ABL:option:{task_id}:{key}",
            lambda t=task_id, k=key, tp=temp: study._measure_option(t, k, tp, study.repository.entries()))
    tracker.write_json("ablations/extra_options.json", extra)
    rescored = part1(cfg, tracker, splits=data.splits, extra_options=extra,
                     unmeasured_file="unmeasured_after_part2.json")
    return {"fusion_summary": summary, "extra_options": {t: sorted(v) for t, v in extra.items()}, "part1": rescored}


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--part", choices=["1", "2", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="continue Part 2 from its checkpoint")
    args = parser.parse_args()
    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    summary: dict[str, Any] = {}
    if args.part in ("1", "all"):
        needs_splits = any((s or {}).get("task_representation")
                           for s in (cfg.section("ablations").get("feature_variants") or {}).values())
        summary["part1"] = part1(cfg, tracker, splits=prepare_data(cfg).splits if needs_splits else None)
    if args.part in ("2", "all"):
        summary["part2"] = part2(cfg, tracker, args)
    tracker.write_json(f"ablations/summary_part{args.part}.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
