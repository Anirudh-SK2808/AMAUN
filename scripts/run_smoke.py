#!/usr/bin/env python
"""End-to-end smoke test: exercise EVERY module once, cheaply.

PHASES 5-7 of the execution order, and the gate idea.md S14 requires before any expensive run.
A full experiment must never be needed to discover that basic functionality is broken, so this
touches every module in the pipeline and reports a pass/fail table.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker

CHECKS: list[tuple[str, str]] = []


def check(name: str):
    def decorator(function):
        def wrapper(state):
            try:
                detail = function(state) or "ok"
                CHECKS.append((name, f"PASS  {detail}"))
                return True
            except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
                CHECKS.append((name, f"FAIL  {type(exc).__name__}: {exc}"))
                state["failures"].append((name, traceback.format_exc()))
                return False
        wrapper.__name__ = function.__name__
        return wrapper
    return decorator


@check("1. config resolution")
def step_config(state):
    cfg = state["cfg"]
    return f"run_mode={cfg.run_mode.value} backbone={cfg.backbone.model_id} hash={cfg.config_hash[:10]}"


@check("2. dataset splits + leakage audit")
def step_data(state):
    data = prepare_data(state["cfg"])
    state["data"] = data
    assert data.leakage_report["clean"], "leakage audit failed"
    return f"{len(data.tasks)} tasks, {data.leakage_report['pairs_checked']} split pairs clean"


@check("3. backbone load + freeze")
def step_backbone(state):
    from amaun.models.backbone import backbone_fingerprint, load_backbone

    backbone = load_backbone(state["cfg"])
    state["backbone"] = backbone
    state["fingerprint"] = backbone_fingerprint(backbone.model)
    trainable = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    assert trainable == 0, f"{trainable} trainable backbone parameters"
    return f"{backbone.frozen_parameter_count} tensors frozen, fp={state['fingerprint'][:10]}"


@check("4. LoRA injection")
def step_lora(state):
    from amaun.adapters.lora import LoRASpec, ensure_lora, trainable_parameter_count
    from amaun.models.backbone import assert_backbone_frozen

    spec = LoRASpec.from_config(state["cfg"].section("lora"))
    state["lora_spec"] = spec
    model = ensure_lora(state["backbone"].model, spec)
    state["peft_model"] = model
    assert_backbone_frozen(model)
    return f"r={spec.r}, {trainable_parameter_count(model)} trainable adapter params"


@check("5. single LoRA training")
def step_train(state):
    from amaun.data.splits import trainer_guard
    from amaun.training.lora_trainer import TrainingConfig, train_adapter

    task = state["data"].tasks[0]
    delta, record = train_adapter(
        backbone=state["backbone"], task=task,
        guard=trainer_guard(state["data"].splits[task.task_id]),
        lora_spec=state["lora_spec"],
        train_cfg=TrainingConfig.from_config(state["cfg"].section("training")),
        resolved=state["cfg"], adapter_id="smoke-temp",
    )
    state["temp_delta"] = delta
    return f"{record.steps} steps, loss={record.final_train_loss:.4f}"


@check("6. backbone unchanged by training")
def step_frozen_after(state):
    from amaun.models.backbone import assert_fingerprint_unchanged

    assert_fingerprint_unchanged(state["backbone"].model, state["fingerprint"], phase="smoke")
    return "fingerprint identical"


@check("7. six adapters trained independently")
def step_bootstrap(state):
    from amaun.data.splits import trainer_guard
    from amaun.training.lora_trainer import TrainingConfig, train_adapter

    train_cfg = TrainingConfig.from_config(state["cfg"].section("training"))
    hashes = {}
    for task in state["data"].tasks:
        delta, _ = train_adapter(
            backbone=state["backbone"], task=task,
            guard=trainer_guard(state["data"].splits[task.task_id]),
            lora_spec=state["lora_spec"], train_cfg=train_cfg,
            resolved=state["cfg"], adapter_id=f"seed-{task.task_id}",
        )
        hashes[task.task_id] = delta
    state["seed_deltas"] = hashes
    unique = {delta.content_hash() for delta in hashes.values()}
    assert len(unique) == len(hashes), "seed adapters are not distinct"
    return f"{len(hashes)} adapters, all distinct"


@check("8. adapter repository lifecycle")
def step_repository(state):
    from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
    from amaun.adapters.repository import AdapterEntry, AdapterRepository

    tracker = state["tracker"]
    repository = AdapterRepository(
        tracker.paths.adapters, backbone_fingerprint=state["backbone"].fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    for task_id, delta in state["seed_deltas"].items():
        task = next(t for t in state["data"].tasks if t.task_id == task_id)
        repository.register(
            AdapterEntry(
                delta=delta,
                metadata=AdapterMetadata(
                    adapter_id=f"seed-{task_id}", task_name=task.task_name,
                    task_type=task.task_type.value, dataset=task.hf_id or f"synthetic:{task_id}",
                    dataset_revision=task.hf_revision,
                    model_version=str(state["cfg"].backbone.model_id),
                    backbone_fingerprint=state["backbone"].fingerprint,
                    run_mode=state["cfg"].run_mode.value, lora_rank=delta.rank,
                    target_modules=list(state["lora_spec"].target_modules),
                    origin=AdapterOrigin.TRAINED.value, tasks_served=[task_id],
                ),
            ),
            bootstrap=True, task_index=0,
        )
    state["repository"] = repository
    return f"{len(repository)} adapters, {repository.memory_bytes()} bytes"


@check("9. candidate isolation")
def step_candidate(state):
    from amaun.adapters.repository import AdapterEntry
    from amaun.adapters.metadata import AdapterMetadata
    from amaun.errors import UnauthorisedRegistrationError

    repository = state["repository"]
    entry = AdapterEntry(
        delta=state["temp_delta"],
        metadata=AdapterMetadata(
            adapter_id="smoke-candidate", task_name="t", task_type="generation",
            dataset="synthetic", dataset_revision=None,
            model_version=str(state["cfg"].backbone.model_id),
            backbone_fingerprint=state["backbone"].fingerprint,
            run_mode=state["cfg"].run_mode.value, lora_rank=state["temp_delta"].rank,
            target_modules=list(state["lora_spec"].target_modules),
        ),
    )
    repository.candidates.put(entry)
    assert "smoke-candidate" not in repository
    try:
        repository.register(entry)
    except UnauthorisedRegistrationError:
        return "unauthorised registration correctly refused"
    raise AssertionError("a candidate was registered without a Decision")


@check("10. task representation (SUPPORT only)")
def step_task_repr(state):
    from amaun.data.splits import manager_guard
    from amaun.manager.task_representation import build_task_representation

    task = state["data"].tasks[0]
    representation = build_task_representation(
        task.task_id, manager_guard(state["data"].splits[task.task_id]),
        config=state["cfg"].section("task_repr"), backbone=state["backbone"],
    )
    state["representation"] = representation
    return f"encoder={representation.encoder}, dim={len(representation.embedding)}"


@check("11. seven feature families")
def step_features(state):
    from amaun.features.extractor import CandidateContext, FeatureExtractor

    extractor = FeatureExtractor(state["cfg"].section("features"))
    state["extractor"] = extractor
    entry = state["repository"].entries()[0]
    vector = extractor.extract(
        CandidateContext(
            temp_id="smoke-temp", temp_delta=state["temp_delta"],
            candidate_id=entry.adapter_id, candidate_delta=entry.delta,
            candidate_metadata=entry.metadata, task_index=0, total_events=1,
            task_embedding=state["representation"].embedding,
            repository_bytes=float(state["repository"].memory_bytes()),
            repository_size=len(state["repository"]),
        )
    )
    state["vector"] = vector
    assert len(extractor.enabled) == 7, f"only {len(extractor.enabled)} families enabled"
    return f"{len(vector)} features over {len(extractor.enabled)} families ({len(vector.missing)} unknown)"


@check("12. normaliser (train-split only)")
def step_normalizer(state):
    from amaun.features.normalizer import FeatureNormalizer

    normalizer = FeatureNormalizer()
    normalizer.fit([state["vector"]] * 4, split="utility_train", run_mode=state["cfg"].run_mode.value)
    state["normalizer"] = normalizer
    normalizer.transform(state["vector"], run_mode=state["cfg"].run_mode.value)
    return f"fitted on '{normalizer.fitted_on_split}' under '{normalizer.fitted_run_mode}'"


@check("13. utility network + decision")
def step_decision(state):
    from amaun.manager.decision import Thresholds
    from amaun.manager.retrieval import score_candidates
    from amaun.manager.utility_network import OUTCOME_HEADS, UtilityNetwork, UtilityNetworkConfig
    from amaun.baselines.strategies import AmaunStrategy

    network = UtilityNetwork(
        len(state["vector"]),
        UtilityNetworkConfig.from_config(state["cfg"].section("utility")["network"]),
    )
    vectors = {entry.adapter_id: state["vector"] for entry in state["repository"].entries()[:2]}
    scores = score_candidates(
        network=network, normalizer=state["normalizer"], vectors=vectors,
        run_mode=state["cfg"].run_mode.value,
    )
    strategy = AmaunStrategy(
        network=network, normalizer=state["normalizer"],
        thresholds=Thresholds(tau_u=-1e9, tau_reuse=1e9, source="fixed", selected_on="smoke"),
        decision_order=str(state["cfg"].get("manager.decision_order")),
        allowed_outcomes=state["cfg"].get("manager.allowed_outcomes"),
        top_k=int(state["cfg"].get("manager.retrieval.k")),
    )
    decision = strategy.decide(
        task_id="smoke", task_index=0, scores=scores,
        repository_size=len(state["repository"]), temp_adapter_trained=True, config_hash="smoke",
    )
    state["decision"] = decision
    return f"outcome={decision.outcome.value}, {len(scores)} candidates scored"


@check("14. layer-wise fusion + KD")
def step_fusion(state):
    from amaun.data.preprocess import collate
    from amaun.data.splits import SplitName, trainer_guard
    from amaun.fusion.layerwise import FusionConfig
    from amaun.fusion.optimize import run_fusion_kd
    from amaun.training.kd import KDConfig
    from amaun.training.lora_trainer import TrainingConfig

    task = state["data"].tasks[0]
    guard = trainer_guard(state["data"].splits[task.task_id])
    batches = [
        collate(
            state["backbone"].tokenizer, guard.get(SplitName.TRAIN)[:2],
            max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
            prompt_truncation=task.prompt_truncation,
        )
    ]
    retention = [
        collate(
            state["backbone"].tokenizer, guard.get(SplitName.VALIDATION)[:2],
            max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
            prompt_truncation=task.prompt_truncation,
        )
    ]
    fusion_cfg = FusionConfig.from_config(
        state["cfg"].section("fusion"),
        lora_learning_rate=TrainingConfig.from_config(state["cfg"].section("training")).learning_rate,
    )
    old = state["repository"].entries()[0].delta
    fused, record = run_fusion_kd(
        backbone=state["backbone"], peft_model=state["peft_model"],
        components=[old, state["temp_delta"]], task_batches=batches, retention_batches=retention,
        fusion_cfg=fusion_cfg, kd_cfg=KDConfig.from_config(state["cfg"].section("kd")),
        fused_adapter_id="smoke-fused", default_rank=state["lora_spec"].r,
    )
    state["fused"] = fused
    state["fusion_record"] = record
    initial = record.alpha_trajectory[0]["alpha"]
    first_layer = next(iter(initial.values()))
    assert abs(first_layer[0] - 0.5) < 1e-6, f"alpha did not start at 0.5: {first_layer}"
    assert record.sources_verified_unmutated, "fusion mutated a source adapter"
    assert record.kd["enabled"], "KD did not run"
    return (
        f"alpha 0.5 -> {record.alpha_summary().get('alpha_mean_component_0', 0):.4f}, "
        f"moved={record.alpha_moved_from_init}, teacher={record.kd['teacher_adapter_id']}, "
        f"{record.trainable_parameters['alpha_parameters']} alpha params"
    )


@check("15. repository compression")
def step_compression(state):
    from amaun.manager.compression import CompressionMechanism, plan_compression, should_compress

    repository = state["repository"]
    triggered, reason = should_compress(repository, repository.memory_bytes())
    assert triggered, reason
    plan, evaluated = plan_compression(
        repository,
        pair_utility=lambda left, right: float(len(left.adapter_id) + len(right.adapter_id)),
        mechanism=CompressionMechanism.UTILITY, min_adapters=2,
    )
    assert plan is not None
    return f"{evaluated} pairs evaluated, would merge {plan.first_id}+{plan.second_id}"


@check("16. evaluation on TEST")
def step_evaluate(state):
    from amaun.adapters.lora import load_delta_into
    from amaun.data.splits import evaluator_guard
    from amaun.evaluation.evaluator import Evaluator
    from amaun.models.generation import GenerationConfig

    task = state["data"].tasks[0]
    load_delta_into(state["peft_model"], state["repository"].entries()[0].delta)
    evaluator = Evaluator(
        state["backbone"], GenerationConfig.from_config(state["cfg"].section("generation")),
        max_examples=2, batch_size=2,
    )
    result = evaluator.evaluate(
        task, evaluator_guard(state["data"].splits[task.task_id]),
        model=state["peft_model"], adapter_id="seed-sum",
    )
    state["eval_result"] = result
    return f"{result.primary_score_name}={result.primary_score:.4f} on {result.n_examples} examples"


@check("17. continual metrics")
def step_cl_metrics(state):
    from amaun.evaluation.cl_metrics import PerformanceMatrix, compute

    matrix = PerformanceMatrix(
        task_order=["a", "b"], zero_shot_baseline={"a": 0.1, "b": 0.1}
    )
    matrix.record("a", "a", 0.8)
    matrix.record("a", "b", 0.2)
    matrix.record("b", "a", 0.6)
    matrix.record("b", "b", 0.7)
    metrics = compute(matrix)
    return (
        f"avg={metrics.average_performance:.3f} forget={metrics.forgetting:.3f} "
        f"bwt={metrics.backward_transfer:.3f} fwt={metrics.forward_transfer:.3f}"
    )


@check("18. federated round")
def step_federated(state):
    from amaun.federated.compatibility import ClientUpdate
    from amaun.federated.server import FederatedServer

    federated_cfg = dict(state["cfg"].section("federated"))
    server = FederatedServer(federated_cfg)
    entries = state["repository"].entries()[:2]
    updates = [
        ClientUpdate(client_id=f"c{i}", entry=entry, n_examples=100 * (i + 1), round_index=0)
        for i, entry in enumerate(entries)
    ]
    results, report = server.run_round(0, updates, target_rank=state["lora_spec"].r)
    ledger = server.ledger.as_dict()
    return (
        f"{len(report.groups)} compatibility groups, {report.payload_bytes_total} bytes, "
        f"{ledger['n_transmissions']} transmissions"
    )


@check("19. baseline strategies")
def step_baselines(state):
    from amaun.baselines.registry import build_strategy, implemented_ids, unimplemented_report
    from amaun.errors import BaselineRequiresCitationError

    built = []
    for baseline_id in ("B2_separate_lora", "B3_naive_average", "B4_similarity_selection"):
        built.append(build_strategy(baseline_id, state["cfg"].section("manager")).strategy_id)
    refused = 0
    for baseline_id in ("LoRM", "FedALoRA", "FedZKD"):
        try:
            build_strategy(baseline_id, {})
        except BaselineRequiresCitationError:
            refused += 1
    assert refused == 3
    return f"{len(built)} built, {refused} citation-gated, {len(implemented_ids())} registered"


@check("20. budget parity gate")
def step_budget(state):
    from amaun.errors import BudgetParityError
    from amaun.training.budget import ArmBudget, BudgetLedger

    ledger = BudgetLedger(tolerance=float(state["cfg"].get("experiment.budget_tolerance")))
    ledger.arms["a"] = ArmBudget(arm_id="a", optimizer_steps=100, tokens_seen=1000, examples_seen=50)
    ledger.arms["b"] = ArmBudget(arm_id="b", optimizer_steps=10, tokens_seen=100, examples_seen=5)
    try:
        ledger.check_parity("a", "b")
    except BudgetParityError:
        return "unequal budgets correctly refused"
    raise AssertionError("budget parity gate did not fire")


@check("21. tracking artefacts")
def step_tracking(state):
    tracker = state["tracker"]
    tracker.log_metric(phase="smoke", check="tracking")
    assert tracker.paths.config_used.exists()
    assert tracker.paths.provenance.exists()
    assert list(tracker.read("metrics"))
    return f"artefacts under {tracker.paths.root}"


STEPS = [
    step_config, step_data, step_backbone, step_lora, step_train, step_frozen_after,
    step_bootstrap, step_repository, step_candidate, step_task_repr, step_features,
    step_normalizer, step_decision, step_fusion, step_compression, step_evaluate,
    step_cl_metrics, step_federated, step_baselines, step_budget, step_tracking,
]


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    state = {"cfg": cfg, "tracker": tracker, "failures": []}

    for step in STEPS:
        ok = step(state)
        if not ok and args.stop_on_failure:
            break

    print(f"\n{'=' * 78}")
    print(f"AMAUN SMOKE TEST  {smoke_label(cfg.run_mode) or '(REPORTABLE MODE)'}")
    print(f"{'=' * 78}")
    for name, outcome in CHECKS:
        print(f"{name:42s} {outcome}")
    passed = sum(1 for _n, outcome in CHECKS if outcome.startswith("PASS"))
    print(f"{'-' * 78}")
    print(f"{passed}/{len(CHECKS)} checks passed")

    tracker.write_json("smoke_report.json", {
        "run_mode": cfg.run_mode.value,
        "checks": [{"name": n, "outcome": o} for n, o in CHECKS],
        "passed": passed,
        "total": len(CHECKS),
        "all_passed": passed == len(CHECKS),
    })

    if state["failures"]:
        print("\nFAILURE DETAIL")
        for name, trace in state["failures"]:
            print(f"\n--- {name} ---\n{trace}")
        return 1
    print("\nSmoke test passed: every module executed at least once.")
    print("Per idea.md S14, expensive runs are unblocked only after this passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
