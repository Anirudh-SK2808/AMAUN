"""M0 -- config resolution, run-mode discipline, quantisation gating (criteria I1, I4, I5)."""

import pytest

from amaun.config import BACKBONE_REGISTRY, RunMode, resolve
from amaun.errors import ConfigError, MissingRunModeError, QuantizationUnavailableError


def test_run_mode_required_no_default():
    with pytest.raises(MissingRunModeError):
        resolve({"experiment_id": "x", "experiment": {"seeds": [0]}})


def test_only_actual_is_reportable():
    assert RunMode.ACTUAL.is_reportable
    assert not RunMode.LOCAL_SMOKE.is_reportable
    assert not RunMode.DUMMY.is_reportable


def test_every_run_mode_has_a_registry_entry():
    for mode in RunMode:
        assert mode in BACKBONE_REGISTRY
    # The dummy tier must be buildable offline, i.e. have no checkpoint id.
    assert BACKBONE_REGISTRY[RunMode.DUMMY].model_id is None
    assert BACKBONE_REGISTRY[RunMode.ACTUAL].model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_hash_is_stable_and_sensitive(raw_config):
    a = resolve(dict(raw_config), run_mode="dummy", experiment_id="x", seed=1)
    b = resolve(dict(raw_config), run_mode="dummy", experiment_id="x", seed=1)
    c = resolve(dict(raw_config), run_mode="dummy", experiment_id="x", seed=2)
    assert a.config_hash == b.config_hash
    assert a.config_hash != c.config_hash


def test_quantization_unavailable_raises_rather_than_downgrading(raw_config):
    """D-008: a silent downgrade would misreport the training recipe."""
    import torch

    if torch.cuda.is_available():  # pragma: no cover - CI hosts vary
        pytest.skip("CUDA present, so nf4 is available")
    with pytest.raises(QuantizationUnavailableError):
        resolve(dict(raw_config), run_mode="actual", experiment_id="x", seed=0)


def test_mandatory_keys_have_no_silent_default(cfg):
    with pytest.raises(ConfigError):
        cfg.get("training.some_key_that_does_not_exist")
    assert cfg.get("training.some_key_that_does_not_exist", 7) == 7


def test_run_mode_overlay_applied(cfg):
    # The dummy overlay caps steps so tests stay fast; it must actually take effect.
    assert cfg.get("training.max_steps") == 2


def test_config_used_records_provenance(cfg):
    payload = cfg.to_config_used()
    assert payload["run_mode"] == "dummy"
    assert payload["is_reportable"] is False
    for key in ("git", "packages", "hardware"):
        assert key in payload["provenance"]
    assert payload["provenance"]["packages"]["torch"]
