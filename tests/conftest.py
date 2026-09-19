"""Shared fixtures.  Every test runs on the ``dummy`` tier: offline, CPU, seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amaun.adapters.delta import AdapterDelta, LayerDelta
from amaun.adapters.lora import LoRASpec
from amaun.adapters.metadata import AdapterMetadata
from amaun.config import load_config, resolve
from amaun.data.providers import synthetic_pool
from amaun.data.registry import load_curriculum
from amaun.data.splits import SplitRatios, build_splits
from amaun.models.backbone import BackboneCache, load_backbone

CONFIG = ROOT / "configs" / "experiments" / "main_bootstrap.yaml"


@pytest.fixture(scope="session")
def raw_config():
    return load_config(CONFIG, config_root=ROOT / "configs")


@pytest.fixture
def cfg(raw_config, tmp_path):
    return resolve(dict(raw_config), run_mode="dummy", experiment_id="pytest")


@pytest.fixture(scope="session")
def session_cfg(raw_config):
    return resolve(dict(raw_config), run_mode="dummy", experiment_id="pytest-session")


@pytest.fixture(scope="session")
def backbone(session_cfg):
    """One tiny frozen backbone for the whole session (invariant I1.4)."""
    BackboneCache.release()
    loaded = load_backbone(session_cfg)
    yield loaded
    BackboneCache.release()


@pytest.fixture(scope="session")
def tasks(session_cfg):
    return load_curriculum(session_cfg.section("datasets"))


@pytest.fixture
def splits(session_cfg, tasks):
    ratios = SplitRatios.from_config(session_cfg.get("data.split_ratios"))
    seed = int(session_cfg.get("data.seed"))
    out = {}
    for task in tasks:
        out[task.task_id] = build_splits(
            task.task_id,
            training_pool=synthetic_pool(task, 64, seed=seed),
            test_pool=synthetic_pool(task, 16, seed=seed, offset=1_000_000),
            seed=seed, ratios=ratios,
        )
    return out


@pytest.fixture
def lora_spec(session_cfg):
    return LoRASpec.from_config(session_cfg.section("lora"))


def make_delta(seed: int, adapter_id: str, *, rank: int = 4, layers: int = 3,
               d_in: int = 8, d_out: int = 6) -> AdapterDelta:
    """A deterministic synthetic adapter delta."""
    generator = torch.Generator().manual_seed(seed)
    return AdapterDelta(
        layers={
            f"l{i}": LayerDelta(
                A=torch.randn(rank, d_in, generator=generator),
                B=torch.randn(d_out, rank, generator=generator),
                scaling=1.5,
            )
            for i in range(layers)
        },
        adapter_id=adapter_id,
    )


def make_metadata(adapter_id: str, **overrides) -> AdapterMetadata:
    base = dict(
        adapter_id=adapter_id, task_name="task", task_type="generation", dataset="synthetic",
        dataset_revision=None, model_version="dummy", backbone_fingerprint="fp",
        run_mode="dummy", lora_rank=4, target_modules=["q_proj"], tasks_served=[adapter_id],
    )
    base.update(overrides)
    return AdapterMetadata(**base)
