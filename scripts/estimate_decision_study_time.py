#!/usr/bin/env python
"""Measure this machine's throughput on the decision study's own code paths and project the
study's wall time (docs/l4_decision_study.md S7 gave only assumed throughputs).

Measured, per task, through the functions the study itself calls:

- training micro-batches       ``train_adapter`` for a few optimiser steps (shuffled TRAIN batches)
- teacher-forced loss batches  ``_validation_loss`` on VALIDATION
- greedy generation            prompt prefill + first token, and the per-token decode cost,
                               on VALIDATION prompts (forced lengths 1 and the generation cap)
- fusion + KD                  ``DecisionStudy._fuse`` at 1 and K steps, for every
                               (new task, old adapter) pair the study fuses
- probes                       ``_build_contexts`` + feature extraction and ``reuse_gate_features``
- preflight                    ``DecisionStudy.preflight`` itself (also the batch-size memory check)

The projection multiplies these by the work the study config implies, counted the way
``DecisionStudy`` caches it: 28 TEST evaluations, 21 VALIDATION loss passes, 15 generation
checks, 12 fusions, 5 trainings.  Generation length is the main unknown, so two scenarios are
reported: ``expected`` (trained adapters stop at the reference length, measured from VALIDATION
targets; zero-shot stops where the backbone stopped here) and ``all_at_cap`` (every generation
runs to its cap, the upper bound).

Leakage: TRAIN / VALIDATION / SUPPORT only.  ``SplitAccessGuard.get`` is patched so any TEST read
raises; VALIDATION prompts stand in for TEST prompts.  Only split *sizes* are taken for TEST.

Writes ``results/<run_mode>/<experiment_id>/time_estimate.json``.  The experiment id defaults to
``l4_time_estimate`` so the study's own results directory is never touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from amaun.adapters.lora import load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterOrigin
from amaun.adapters.repository import AdapterEntry
from amaun.continual.decision_study import DecisionStudy
from amaun.data.loaders import BatchPlan, GuardedLoader
from amaun.data.preprocess import _encode, encode_prompt
from amaun.data.splits import SplitAccessGuard, SplitName, trainer_guard, utility_guard
from amaun.errors import SplitAccessViolation
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.manager.bootstrap import _validation_loss
from amaun.manager.reuse_gate import ReuseGateConfig
from amaun.models.backbone import load_backbone
from amaun.training.lora_trainer import train_adapter
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker

DEFAULT_EXPERIMENT_ID = "l4_time_estimate"

_original_get = SplitAccessGuard.get


def _no_test_get(self: SplitAccessGuard, name: SplitName | str):
    if SplitName(name) is SplitName.TEST:
        raise SplitAccessViolation(f"the time estimate must not read TEST (task '{self.task_id}')")
    return _original_get(self, name)


def _timed(device: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--train-steps", type=int, default=8, help="optimiser steps timed per task")
    parser.add_argument("--fusion-steps", type=int, default=6, help="K in the 1-vs-K fusion timing")
    parser.add_argument("--loss-examples", type=int, default=40, help="VALIDATION examples per loss timing")
    parser.add_argument("--gen-examples", type=int, default=8, help="VALIDATION prompts per generation timing")
    args = parser.parse_args()
    if args.experiment_id is None:
        args.experiment_id = DEFAULT_EXPERIMENT_ID

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg)
    if data.blocked:
        print(f"ERROR: blocked tasks {data.blocked}")
        return 2
    test_sizes = {t: s.counts()[SplitName.TEST.value] for t, s in data.splits.items()}
    SplitAccessGuard.get = _no_test_get          # from here on, a TEST read raises

    backbone = load_backbone(cfg)
    study = DecisionStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker)
    device, tokenizer, model = backbone.device, backbone.tokenizer, study.peft_model
    train_cfg = study.runner.train_cfg
    bs, accum = train_cfg.batch_size, train_cfg.gradient_accumulation_steps
    init_tasks = [a.task_id for a in study.study.initial_adapters]
    eval_tasks = [t.task_id for t in study.study.evaluation_tasks]
    all_tasks = list(dict.fromkeys(init_tasks + eval_tasks))
    generation = study.final_evaluator.generation
    gate_section = cfg.section("utility").get("reuse_gate")
    gate_cfg = ReuseGateConfig.from_config(gate_section) if gate_section else None
    wall_start = time.perf_counter()

    def val_examples(task_id: str) -> list:
        return list(utility_guard(study.splits[task_id]).get(SplitName.VALIDATION))

    # -- preflight (also warms CUDA kernels and the allocator) --------------------------
    print("timing preflight ...", flush=True)
    preflight, preflight_seconds = _timed(device, study.preflight)
    if not preflight["ok"]:
        failed = [c for c in preflight["checks"] if not c["ok"]]
        tracker.write_json("time_estimate.json", {"aborted": "preflight OOM", "preflight": preflight})
        print(f"ABORTED: preflight failed at the configured batch size: {failed}")
        return 3

    # -- task representations ------------------------------------------------------------
    representation_seconds = {}
    for task_id in all_tasks:
        _, representation_seconds[task_id] = _timed(device, study.runner.ensure_representation, task_id)

    # -- training micro-batches ----------------------------------------------------------
    probe_train_cfg = dataclasses.replace(train_cfg, max_steps=args.train_steps, eval_every_steps=None,
                                          early_stopping_patience=None)
    warm_cfg = dataclasses.replace(probe_train_cfg, max_steps=1)
    train_adapter(backbone=backbone, task=study.tasks[all_tasks[0]], guard=trainer_guard(study.splits[all_tasks[0]]),
                  lora_spec=study.runner.lora_spec, train_cfg=warm_cfg, resolved=cfg, adapter_id="estimate-warmup")
    training, deltas, entries = {}, {}, {}
    for task_id in all_tasks:
        print(f"timing training on {task_id} ...", flush=True)
        task = study.tasks[task_id]
        delta, record = train_adapter(
            backbone=backbone, task=task, guard=trainer_guard(study.splits[task_id]),
            lora_spec=study.runner.lora_spec, train_cfg=probe_train_cfg, resolved=cfg,
            adapter_id=f"estimate-{task_id}",
        )
        micro = record.steps * accum
        training[task_id] = {
            "micro_batches_timed": micro,
            "seconds_per_micro_batch": record.wall_seconds / max(1, micro),
            "tokens_per_micro_batch": record.tokens_seen / max(1, micro),
            "peak_allocated_bytes": record.peak_memory_bytes,
        }
        deltas[task_id] = delta
        entries[task_id] = AdapterEntry(delta=delta, metadata=study.runner._metadata_for(
            delta.adapter_id, task, delta, record.as_dict(), AdapterOrigin.TRAINED, [task_id]))

    # -- teacher-forced loss batches -----------------------------------------------------
    forward = {}
    for task_id in all_tasks:
        n = min(args.loss_examples, len(val_examples(task_id)))
        _, seconds = _timed(device, _validation_loss, model, deltas[task_id], study.tasks[task_id],
                            study.splits[task_id], tokenizer, device, max_examples=n, batch_size=bs)
        forward[task_id] = {"examples": n, "seconds_per_batch": seconds / max(1, math.ceil(n / bs))}

    # -- generation ----------------------------------------------------------------------
    base_kwargs = generation.to_kwargs(pad_token_id=getattr(tokenizer, "pad_token_id", None),
                                       eos_token_id=getattr(tokenizer, "eos_token_id", None))
    gen = {}
    for task_id in all_tasks:
        print(f"timing generation on {task_id} ...", flush=True)
        task = study.tasks[task_id]
        cap = max(1, min(int(base_kwargs["max_new_tokens"]), task.max_target_tokens))
        examples = val_examples(task_id)
        reference_lengths = [min(cap, len(_encode(tokenizer, e.target)[: task.max_target_tokens]) + 1)
                             for e in examples]

        def run(input_ids: torch.Tensor, *, min_new: int | None, max_new: int) -> tuple[int, float]:
            kwargs = {**base_kwargs, "max_new_tokens": max_new}
            if min_new:
                kwargs["min_new_tokens"] = min_new
            with torch.inference_mode():
                out, seconds = _timed(device, model.generate, input_ids=input_ids,
                                      attention_mask=torch.ones_like(input_ids), **kwargs)
            return int(out.shape[1] - input_ids.shape[1]), seconds

        first, per_token, zero_shot_lengths, prompt_lengths = [], [], [], []
        for example in examples[: args.gen_examples]:
            ids = encode_prompt(tokenizer, example.prompt, max_input_tokens=task.max_input_tokens,
                                truncation=task.prompt_truncation)
            prompt_lengths.append(len(ids))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            load_delta_into(model, deltas[task_id])
            _, t1 = run(input_ids, min_new=1, max_new=1)
            n_cap, t_cap = run(input_ids, min_new=cap, max_new=cap)
            first.append(t1)
            if n_cap > 1:
                per_token.append((t_cap - t1) / (n_cap - 1))
            zero_delta_(model)
            n_natural, _ = run(input_ids, min_new=None, max_new=cap)
            zero_shot_lengths.append(n_natural)
        gen[task_id] = {
            "cap": cap,
            "prompts_timed": len(first),
            "mean_prompt_tokens": _mean(prompt_lengths),
            "seconds_first_token": _mean(first),
            "seconds_per_extra_token": _mean(per_token),
            "expected_length_trained": _mean(reference_lengths),
            "observed_length_zero_shot": _mean(zero_shot_lengths),
            "reference_length_source": f"{len(reference_lengths)} VALIDATION targets, +1 for EOS, capped",
        }

    # -- probes ----------------------------------------------------------------------------
    def probe_seconds(task_id: str, candidate_ids: list[str], index: int) -> dict[str, float]:
        candidates = [entries[c] for c in candidate_ids]

        def contexts() -> None:
            built = study.runner._build_contexts(task_id, index, deltas[task_id], candidates=candidates)
            for context in built.values():
                study.runner.extractor.extract(context)

        _, ctx = _timed(device, contexts)
        gate = 0.0
        if gate_cfg is not None:
            _, gate = _timed(device, study.runner.reuse_gate_features, task_id, candidates,
                             probe_examples=gate_cfg.probe_examples)
        return {"contexts": ctx, "gate": gate}

    probes_phase2 = {t: probe_seconds(t, [next(o for o in init_tasks if o != t)], 0) for t in init_tasks}
    probes_phase3 = {t: probe_seconds(t, init_tasks, i) for i, t in enumerate(eval_tasks, start=1)}

    # -- fusion + KD -----------------------------------------------------------------------
    pairs = list(itertools.permutations(init_tasks, 2)) + [(e, j) for e in eval_tasks for j in init_tasks]
    fusion = {}
    k = max(2, args.fusion_steps)
    for new, old in pairs:
        print(f"timing fusion {new} into {old} ...", flush=True)
        _, t1 = _timed(device, study._fuse, entries[old], deltas[new], new, f"estimate-{new}-into-{old}-1", steps=1)
        _, tk = _timed(device, study._fuse, entries[old], deltas[new], new, f"estimate-{new}-into-{old}-{k}", steps=k)
        step = max(0.0, (tk - t1) / (k - 1))
        fusion[f"{new}->{old}"] = {"seconds_per_step": step, "fixed_seconds": max(0.0, t1 - step)}

    # -- projection --------------------------------------------------------------------------
    n_test = int(cfg.get("evaluation.max_test_examples"))
    n_loss = study.study.supervision_loss_examples
    n_check = study.study.supervision_generation_examples
    fusion_steps = study.runner.fusion_cfg.steps
    n_val = {t: len(val_examples(t)) for t in all_tasks}

    def generate_seconds(task_id: str, length: float) -> float:
        g = gen[task_id]
        return g["seconds_first_token"] + g["seconds_per_extra_token"] * max(0.0, length - 1)

    def eval_seconds(task_id: str, n: int, length: float) -> float:
        return math.ceil(n / bs) * forward[task_id]["seconds_per_batch"] + n * generate_seconds(task_id, length)

    def fuse_seconds(new: str, old: str, steps: int) -> float:
        f = fusion[f"{new}->{old}"]
        return f["fixed_seconds"] + steps * f["seconds_per_step"]

    def train_seconds(task_id: str) -> float:
        loader = GuardedLoader(trainer_guard(study.splits[task_id]), SplitName.TRAIN, study.tasks[task_id],
                               tokenizer, BatchPlan(batch_size=bs, shuffle=True, seed=cfg.seed))
        steps = train_cfg.max_steps or max(1, int(max(1, len(loader) // accum) * train_cfg.epochs))
        evals = steps // train_cfg.eval_every_steps if train_cfg.eval_every_steps else 0
        per_eval = (math.ceil(min(n_val[task_id], train_cfg.intermediate_eval_examples) / bs)
                    if train_cfg.intermediate_eval_examples else 8)
        return (steps * accum * training[task_id]["seconds_per_micro_batch"]
                + evals * per_eval * forward[task_id]["seconds_per_batch"])

    def project(scenario: str) -> dict[str, float]:
        def trained(t: str) -> float:
            return gen[t]["cap"] if scenario == "all_at_cap" else gen[t]["expected_length_trained"]

        def zero_shot(t: str) -> float:
            return gen[t]["cap"] if scenario == "all_at_cap" else gen[t]["observed_length_zero_shot"]

        def test_eval(t: str, length: float) -> float:
            return eval_seconds(t, min(n_test, test_sizes[t]), length)

        phases: dict[str, float] = {"preflight": preflight_seconds}
        phases["1_train_initial_adapters"] = sum(train_seconds(t) + representation_seconds[t] for t in init_tasks)
        phases["1_standalone_test"] = sum(test_eval(t, trained(t)) for t in init_tasks)
        if study.study.evaluate_zero_shot:
            phases["1_zero_shot_test"] = sum(test_eval(t, zero_shot(t)) for t in all_tasks)

        pair_list = list(itertools.permutations(init_tasks, 2))
        standalone_losses = {(j, i) for i, j in pair_list} | {(i, i) for i, _ in pair_list} | {(j, j) for _, j in pair_list}
        loss_pass = {t: math.ceil(min(n_loss, n_val[t]) / bs) * forward[t]["seconds_per_batch"] for t in init_tasks}
        check = {t: eval_seconds(t, min(n_check, n_val[t]), trained(t)) for t in init_tasks}
        phases["2_supervision_probes"] = sum(probes_phase2[i]["contexts"] + probes_phase2[i]["gate"] for i, _ in pair_list)
        phases["2_supervision_fusion"] = sum(fuse_seconds(i, j, fusion_steps) for i, j in pair_list)
        phases["2_supervision_val_loss"] = (sum(loss_pass[task] for _, task in standalone_losses)
                                            + sum(loss_pass[i] + loss_pass[j] for i, j in pair_list))
        phases["2_supervision_generation_checks"] = (sum(check[i] for i in init_tasks)
                                                     + sum(check[i] + check[j] for i, j in pair_list))

        for e in eval_tasks:
            phases[f"3_{e}_train_temp"] = train_seconds(e) + representation_seconds[e]
            phases[f"3_{e}_probes"] = probes_phase3[e]["contexts"] + probes_phase3[e]["gate"]
            phases[f"3_{e}_fusion"] = sum(fuse_seconds(e, j, fusion_steps) for j in init_tasks)
            # CREATE + REUSE_j + FUSE_j on the new task; FUSE_j also on its old task.
            phases[f"3_{e}_test_evals"] = ((1 + 2 * len(init_tasks)) * test_eval(e, trained(e))
                                           + sum(test_eval(j, trained(j)) for j in init_tasks))

        cycles = study.study.memory_cycles
        if cycles and len(init_tasks) >= 2:
            first_task, second_task = init_tasks[0], init_tasks[1]
            n_audit = min(study.study.memory_cycle_eval_examples, n_val[first_task])
            phases["4_memory_audit"] = cycles * (
                2 * eval_seconds(first_task, n_audit, trained(first_task))
                + fuse_seconds(first_task, second_task, study.study.memory_cycle_fusion_steps))
        phases["total"] = sum(phases.values())
        return phases

    budget_seconds = study.study.time_budget_hours * 3600
    projections = {s: project(s) for s in ("expected", "all_at_cap")}
    report = {
        "run_mode": cfg.run_mode.value,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "config": args.config,
        "batch_size": bs, "gradient_accumulation_steps": accum,
        "time_budget_hours": study.study.time_budget_hours,
        "measurements": {
            "preflight": {"seconds": preflight_seconds, "checks": preflight["checks"]},
            "representation_seconds": representation_seconds,
            "training": training, "forward": forward, "generation": gen,
            "probes_phase2": probes_phase2, "probes_phase3": probes_phase3, "fusion": fusion,
        },
        "projection_seconds": projections,
        "fits_time_budget": {s: p["total"] <= budget_seconds for s, p in projections.items()},
        "estimator_wall_seconds": time.perf_counter() - wall_start,
        "not_measured": [
            "utility-network / reuse-gate fitting (CPU, six examples)",
            "adapter checkpoint writes and JSON logging",
            "metric computation on generated text (CPU)",
        ],
        "caveats": [
            "VALIDATION prompts stand in for TEST prompts (TEST is never read)",
            "REUSE options are projected with the new task's reference length",
            "the adapters timed here are barely trained; cost per token does not depend on that, "
            "but zero-shot stopping lengths are measured on the frozen backbone",
        ],
    }
    path = tracker.write_json("time_estimate.json", report)

    print(f"\n{smoke_label(cfg.run_mode)}  device={device}  gpu={report['gpu']}")
    print(f"{'task':<10} {'s/micro':>8} {'tok/micro':>9} {'s/fwd':>7} {'1st tok s':>9} {'ms/tok':>7} "
          f"{'cap':>4} {'L_trained':>9} {'L_zero':>7}")
    for t in all_tasks:
        g = gen[t]
        print(f"{t:<10} {training[t]['seconds_per_micro_batch']:>8.3f} {training[t]['tokens_per_micro_batch']:>9.0f} "
              f"{forward[t]['seconds_per_batch']:>7.3f} {g['seconds_first_token']:>9.3f} "
              f"{1000 * g['seconds_per_extra_token']:>7.1f} {g['cap']:>4} {g['expected_length_trained']:>9.1f} "
              f"{g['observed_length_zero_shot']:>7.1f}")
    print(f"\n{'phase':<36} {'expected h':>10} {'all-at-cap h':>13}")
    for phase in projections["expected"]:
        print(f"{phase:<36} {projections['expected'][phase] / 3600:>10.2f} "
              f"{projections['all_at_cap'][phase] / 3600:>13.2f}")
    print(f"\nbudget {study.study.time_budget_hours} h; fits: {report['fits_time_budget']}")
    print(f"estimator took {report['estimator_wall_seconds'] / 60:.1f} min; report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
