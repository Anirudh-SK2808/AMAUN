"""D-026: merged-weight, batched evaluation (l4_full_study), and the projection fixes that go with it.

Merging computes the same function as the installed LoRA branch (exactly, up to float rounding) and
must leave the backbone bit-for-bit as it was; batched generation must return each example's own
prediction, in order.  On the dummy tier (fp32) both are checked against the run-1 paths directly.
"""

import dataclasses
from pathlib import Path
import types

import torch

from amaun.adapters.delta import AdapterDelta, LayerDelta
from amaun.adapters.lora import MergedWeights, ensure_lora, iter_lora_modules, load_delta_into, zero_delta_
from amaun.config import load_config, resolve
from amaun.data.splits import evaluator_guard
from amaun.evaluation.evaluator import Evaluator
from amaun.models.backbone import assert_fingerprint_unchanged
from amaun.models.generation import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]


def _random_delta(model, *, seed: int) -> AdapterDelta:
    generator = torch.Generator().manual_seed(seed)
    layers = {}
    for key, module in iter_lora_modules(model):
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        layers[key] = LayerDelta(A=0.1 * torch.randn(a.shape, generator=generator),
                                 B=0.1 * torch.randn(b.shape, generator=generator), scaling=2.0)
    return AdapterDelta(layers=layers, adapter_id=f"random-{seed}")


def _ids(backbone):
    return torch.tensor([[5, 17, 42, 9, 3, 77, 120, 64]], dtype=torch.long, device=backbone.device)


def test_merged_weights_restore_the_backbone_exactly(backbone, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    merger = MergedWeights(model)
    delta = _random_delta(model, seed=1)
    base = {k: m.base_layer.weight.detach().clone() for k, m in merger.modules.items()}
    lora_b = {k: m.lora_B["default"].weight.detach().clone() for k, m in merger.modules.items()}
    grads = {name: p.requires_grad for name, p in model.named_parameters()}
    with merger.apply(delta):
        assert any(not torch.equal(base[k], m.base_layer.weight) for k, m in merger.modules.items())
    assert all(torch.equal(base[k], m.base_layer.weight) for k, m in merger.modules.items())
    assert all(torch.equal(lora_b[k], m.lora_B["default"].weight) for k, m in merger.modules.items())
    assert {name: p.requires_grad for name, p in model.named_parameters()} == grads
    assert not any(m.disable_adapters for m in merger.modules.values())
    assert_fingerprint_unchanged(backbone.model, backbone.fingerprint, phase="merged-weights test")


@torch.no_grad()
def test_merged_weights_compute_the_installed_lora_function(backbone, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    model.eval()
    merger = MergedWeights(model)
    delta, ids = _random_delta(model, seed=2), _ids(backbone)
    zero_delta_(model)
    plain = model(input_ids=ids).logits
    load_delta_into(model, delta)
    installed = model(input_ids=ids).logits
    with merger.apply(delta) as merged_model:
        merged = merged_model(input_ids=ids).logits
    zero_delta_(model)
    assert not torch.allclose(installed, plain, atol=1e-3)          # the delta has an effect
    assert torch.allclose(merged, installed, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_merged_backbone_ignores_whatever_lora_factors_are_installed(backbone, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    model.eval()
    ids = _ids(backbone)
    zero_delta_(model)
    plain = model(input_ids=ids).logits
    load_delta_into(model, _random_delta(model, seed=3))
    with MergedWeights(model).apply(None) as backbone_only:
        assert torch.equal(backbone_only(input_ids=ids).logits, plain)
    zero_delta_(model)


def test_batched_generation_returns_each_examples_own_prediction_in_order(session_cfg, backbone, tasks,
                                                                          splits, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    zero_delta_(model)
    one = GenerationConfig.from_config(session_cfg.section("generation"))
    four = dataclasses.replace(one, batch_size=4)
    for task in tasks[:2]:
        guard = evaluator_guard(splits[task.task_id])
        runs = [Evaluator(backbone, generation, batch_size=2).evaluate(
            task, guard, model=model, adapter_id=None, max_examples=7) for generation in (one, four)]
        assert [r["prediction"] for r in runs[0].per_example] == [r["prediction"] for r in runs[1].per_example]
        assert [r["example_id"] for r in runs[0].per_example] == [r["example_id"] for r in runs[1].per_example]
        assert runs[1].generation_config["batch_size"] == 4


def _full_study_workload():
    from amaun.continual.decision_study import StudyConfig
    from amaun.continual.full_study import FullStudy, FullStudyConfig
    from amaun.experiment import prepare_data
    from amaun.training.lora_trainer import TrainingConfig

    raw = load_config(ROOT / "configs" / "experiments" / "l4_full_study.yaml", config_root=ROOT / "configs")
    cfg = resolve(raw, run_mode="dummy", experiment_id="pytest-workload")
    data = prepare_data(cfg)
    study = FullStudy.__new__(FullStudy)
    study.cfg, study.splits = cfg, data.splits
    study.tasks = {t.task_id: t for t in data.tasks}
    study.study = StudyConfig.from_config(cfg)
    train_cfg = TrainingConfig.from_config(cfg.section("training"))
    study.runner = types.SimpleNamespace(train_cfg=train_cfg)
    study.full = FullStudyConfig.from_config(cfg, lora_learning_rate=train_cfg.learning_rate)
    return study, study._workload()


def test_projection_counts_no_generation_for_cached_always_separate_streams():
    study, items = _full_study_workload()
    stream = study.full.stream
    deciding_runs = sum(1 for p in stream["policies"] if p != "always_separate") * len(stream["caps"])
    arrivals = [i for i in items if i.phase == "P6" and i.note == "arrival TEST (upper bound)"]
    assert len(arrivals) == deciding_runs * len(stream["order"])


def test_projection_prices_fuse_old_tasks_at_every_seed_task():
    study, items = _full_study_workload()
    seeds = [a.task_id for a in study.study.initial_adapters]
    old = [i for i in items if i.phase == "P5" and i.note.startswith("old-task TEST of FUSE options")]
    assert sorted({i.task for i in old}) == sorted(seeds)
    assert len(old) == len(seeds) * len(study.study.evaluation_tasks)


def test_a_rank_change_inside_inference_mode_leaves_installable_parameters(backbone, lora_spec):
    """_validation_loss runs under inference_mode; a rank-2r install there must not leave inference
    tensors behind (the merged path no longer installs fused deltas before it)."""
    from amaun.adapters.lora import reset_lora_parameters

    model = ensure_lora(backbone.model, lora_spec)
    wide = _random_delta(model, seed=4)
    wide = AdapterDelta(layers={k: LayerDelta(A=torch.cat([l.A, l.A]), B=torch.cat([l.B, l.B], dim=1),
                                              scaling=l.scaling) for k, l in wide.items()},
                        adapter_id="rank-2r")
    with torch.inference_mode():
        load_delta_into(model, wide)
    load_delta_into(model, wide)                      # in place, outside inference mode
    reset_lora_parameters(model, seed=0)
    zero_delta_(model)
