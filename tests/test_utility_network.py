"""M7.1 -- the utility network (criterion E10)."""

import pytest
import torch

from amaun.errors import ManagerError
from amaun.manager.utility_network import (
    HEAD_ORIENTATION,
    OUTCOME_HEADS,
    UtilityNetwork,
    UtilityNetworkConfig,
    train_utility_network,
)


def _network(input_dim=12, monotone=True):
    cfg = UtilityNetworkConfig(
        hidden_dims=(16, 16), dropout=0.0, activation="relu",
        output_heads=OUTCOME_HEADS, monotone_weights=monotone,
    )
    return UtilityNetwork(input_dim, cfg)


def test_all_six_outcome_heads_are_present():
    """idea.md S5.1: u_hat is the 6-vector of predicted post-integration outcomes."""
    network = _network()
    assert network.heads == OUTCOME_HEADS
    utility, outcomes = network(torch.randn(3, 12))
    assert outcomes.shape == (3, 6)
    assert utility.shape == (3,)


def test_scalarisation_weights_are_nonnegative():
    network = _network()
    with torch.no_grad():
        network.scalar_logits.copy_(torch.tensor([-5.0, -5.0, 3.0, -1.0, 0.0, 2.0]))
    assert bool((network.scalar_weights() >= 0).all())


def test_utility_is_monotone_in_every_oriented_head():
    """Invariant I7.4 -- without this the network could learn to PREFER forgetting."""
    network = _network()
    with torch.no_grad():
        network.scalar_logits.copy_(torch.randn(6))
        base = network.predict_outcomes(torch.randn(1, 12))
        base_utility = float(network.psi(base) @ network.scalar_weights())
        for index, head in enumerate(network.heads):
            bumped = base.clone()
            bumped[0, index] += 1.0
            bumped_utility = float(network.psi(bumped) @ network.scalar_weights())
            if HEAD_ORIENTATION[head] > 0:
                assert bumped_utility >= base_utility - 1e-9, f"{head} should increase utility"
            else:
                assert bumped_utility <= base_utility + 1e-9, f"{head} should decrease utility"


def test_unknown_head_without_orientation_is_rejected():
    with pytest.raises(ManagerError):
        UtilityNetworkConfig.from_config(
            {"hidden_dims": [8], "output_heads": ["mystery_metric"]}
        )


def test_training_reduces_the_regression_loss():
    """idea.md S9: L_U = sum ||g_omega(z) - y||^2."""
    torch.manual_seed(0)
    network = _network()
    features = torch.randn(64, 12)
    # A learnable linear relationship, so a failure to fit indicates a wiring problem.
    targets = features[:, :6] * 0.5
    record = train_utility_network(
        network, features, targets,
        config={"epochs": 60, "batch_size": 16, "learning_rate": 5e-3, "weight_decay": 0.0, "loss": "l2"},
        fit_split="utility_train", run_mode="dummy",
    )
    assert record.final_loss < record.loss_trajectory[0]
    assert set(record.per_head_final_mse) == set(OUTCOME_HEADS)


def test_training_on_test_split_is_refused():
    """idea.md S16.6."""
    network = _network()
    with pytest.raises(ManagerError, match="test"):
        train_utility_network(
            network, torch.randn(8, 12), torch.randn(8, 6),
            config={"epochs": 1, "batch_size": 4, "learning_rate": 1e-3, "loss": "l2"},
            fit_split="test", run_mode="dummy",
        )


def test_target_shape_mismatch_is_refused():
    network = _network()
    with pytest.raises(ManagerError):
        train_utility_network(
            network, torch.randn(8, 12), torch.randn(8, 3),
            config={"epochs": 1, "batch_size": 4, "learning_rate": 1e-3, "loss": "l2"},
            fit_split="utility_train", run_mode="dummy",
        )


def test_unimplemented_loss_is_refused():
    """idea.md S9 allows replacing L2 but requires the choice to be reported."""
    network = _network()
    with pytest.raises(ManagerError):
        train_utility_network(
            network, torch.randn(8, 12), torch.randn(8, 6),
            config={"epochs": 1, "batch_size": 4, "learning_rate": 1e-3, "loss": "huber"},
            fit_split="utility_train", run_mode="dummy",
        )


def test_save_load_rejects_cross_run_mode(tmp_path):
    network = _network()
    path = network.save(tmp_path / "n.pt", run_mode="dummy", feature_names=[f"f{i}" for i in range(12)])
    reloaded, names = UtilityNetwork.load(path, expected_run_mode="dummy")
    assert len(names) == 12
    assert torch.allclose(reloaded.scalar_logits, network.scalar_logits)
    with pytest.raises(ManagerError):
        UtilityNetwork.load(path, expected_run_mode="actual")
