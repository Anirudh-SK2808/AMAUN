"""M7.1 -- the AMAUN utility network (idea.md S5.1).

    h1 = ReLU(W1 z + b1)
    h2 = ReLU(W2 h1 + b2)
    o  = W_o h2 + b_o
    u_hat = [s_post, F_hat, c_hat, Mem_hat, Comm_hat, L_hat]^T
    U  = w^T psi(u_hat)

Two properties are enforced rather than hoped for:

**Monotonicity (invariant I7.4).**  ``psi`` orients every head so larger means better --
forgetting, memory, communication, and loss are negated -- and ``w`` is softplus-parameterised
so its entries are non-negative.  Without this, gradient descent could learn a *negative*
weight on the forgetting head, and the network would then prefer candidates that forget more
while still reporting a high "utility".  That failure would be invisible in the scalar
utility and would quietly invert the method's premise.

**Multi-output supervision (idea.md S9).**  The heads are trained against *observed*
post-integration outcomes, so ``U`` is grounded in measured consequences rather than in a
hand-tuned score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from amaun.errors import ManagerError

# Head order is fixed and must match the target vector's order everywhere.
OUTCOME_HEADS = ("primary_score", "forgetting", "confidence", "memory", "communication", "long_term_loss")

# +1 means "larger is better", -1 means "larger is worse" and is negated by psi.
HEAD_ORIENTATION: dict[str, float] = {
    "primary_score": +1.0,
    "forgetting": -1.0,
    "confidence": +1.0,
    "memory": -1.0,
    "communication": -1.0,
    "long_term_loss": -1.0,
}


@dataclass
class UtilityNetworkConfig:
    hidden_dims: tuple[int, ...]
    dropout: float
    activation: str
    output_heads: tuple[str, ...]
    monotone_weights: bool

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "UtilityNetworkConfig":
        heads = tuple(section.get("output_heads") or OUTCOME_HEADS)
        unknown = [head for head in heads if head not in HEAD_ORIENTATION]
        if unknown:
            raise ManagerError(
                f"utility heads {unknown} have no declared orientation; every head must state "
                f"whether larger is better, or monotonicity cannot be enforced"
            )
        return cls(
            hidden_dims=tuple(int(d) for d in section["hidden_dims"]),
            dropout=float(section.get("dropout", 0.0)),
            activation=str(section.get("activation", "relu")),
            output_heads=heads,
            monotone_weights=bool(section.get("monotone_weights", True)),
        )


class UtilityNetwork(nn.Module):
    """MLP with a multi-objective head and a monotone scalarisation."""

    def __init__(self, input_dim: int, cfg: UtilityNetworkConfig) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ManagerError("utility network input dimension must be positive")
        self.cfg = cfg
        self.input_dim = input_dim
        self.heads = tuple(cfg.output_heads)

        activation = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(cfg.activation)
        if activation is None:
            raise ManagerError(f"unknown activation '{cfg.activation}'")

        layers: list[nn.Module] = []
        width = input_dim
        for hidden in cfg.hidden_dims:
            layers += [nn.Linear(width, hidden), activation()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, len(self.heads))

        # Scalarisation weights.  Softplus keeps them non-negative (invariant I7.4).
        self.scalar_logits = nn.Parameter(torch.zeros(len(self.heads)))
        self.register_buffer(
            "orientation",
            torch.tensor([HEAD_ORIENTATION[head] for head in self.heads], dtype=torch.float32),
        )

    # -- forward -----------------------------------------------------------------
    def predict_outcomes(self, z: torch.Tensor) -> torch.Tensor:
        """``u_hat`` -- the raw predicted post-integration outcomes."""
        return self.head(self.trunk(z))

    def scalar_weights(self) -> torch.Tensor:
        if not self.cfg.monotone_weights:
            return self.scalar_logits
        return torch.nn.functional.softplus(self.scalar_logits)

    def psi(self, outcomes: torch.Tensor) -> torch.Tensor:
        """Orient every head so larger is better, then squash to a comparable range.

        ``tanh`` bounds each oriented outcome, so one head with a wild scale (memory in bytes)
        cannot dominate the scalar utility purely through its units.
        """
        return torch.tanh(outcomes * self.orientation)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(utility, predicted_outcomes)``."""
        outcomes = self.predict_outcomes(z)
        utility = self.psi(outcomes) @ self.scalar_weights()
        return utility, outcomes

    def utility(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)[0]

    # -- persistence -------------------------------------------------------------
    def save(self, path: Path | str, *, run_mode: str, feature_names: Sequence[str]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "config": {
                    "hidden_dims": list(self.cfg.hidden_dims),
                    "dropout": self.cfg.dropout,
                    "activation": self.cfg.activation,
                    "output_heads": list(self.cfg.output_heads),
                    "monotone_weights": self.cfg.monotone_weights,
                },
                "run_mode": run_mode,
                "feature_names": list(feature_names),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, expected_run_mode: str) -> tuple["UtilityNetwork", list[str]]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("run_mode") != expected_run_mode:
            raise ManagerError(
                f"refusing to load a utility network trained under run_mode="
                f"'{payload.get('run_mode')}' into a '{expected_run_mode}' run: the feature "
                f"distribution differs with the backbone (idea.md S9)"
            )
        cfg = UtilityNetworkConfig(
            hidden_dims=tuple(payload["config"]["hidden_dims"]),
            dropout=payload["config"]["dropout"],
            activation=payload["config"]["activation"],
            output_heads=tuple(payload["config"]["output_heads"]),
            monotone_weights=payload["config"]["monotone_weights"],
        )
        model = cls(payload["input_dim"], cfg)
        model.load_state_dict(payload["state_dict"])
        return model, list(payload.get("feature_names", []))


@dataclass
class UtilityTrainingRecord:
    epochs: int = 0
    loss_trajectory: list[float] = field(default_factory=list)
    n_samples: int = 0
    final_loss: float | None = None
    per_head_final_mse: dict[str, float] = field(default_factory=dict)
    fit_split: str = ""
    run_mode: str = ""

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def train_utility_network(
    model: UtilityNetwork,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: Mapping[str, Any],
    fit_split: str,
    run_mode: str,
    seed: int = 0,
) -> UtilityTrainingRecord:
    """Fit ``g_omega`` with ``L_U = sum_i ||g_omega(z_i) - y_i||^2`` (idea.md S9)."""
    if fit_split == "test":
        raise ManagerError(
            "refusing to train the utility network on the test sequence (idea.md S16.6)"
        )
    if features.shape[0] != targets.shape[0]:
        raise ManagerError("feature/target count mismatch")
    if targets.shape[1] != len(model.heads):
        raise ManagerError(
            f"target vector has {targets.shape[1]} columns but the network has "
            f"{len(model.heads)} heads {model.heads}"
        )

    torch.manual_seed(seed)
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loss_name = str(config.get("loss", "l2"))
    if loss_name != "l2":
        raise ManagerError(
            f"utility loss '{loss_name}' is not implemented; idea.md S9 specifies L2 and "
            f"requires any replacement to be reported explicitly"
        )

    record = UtilityTrainingRecord(
        n_samples=int(features.shape[0]), fit_split=fit_split, run_mode=run_mode
    )
    model.train()
    count = features.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(count)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, count, batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted = model.predict_outcomes(features[index])
            loss = torch.nn.functional.mse_loss(predicted, targets[index])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        record.loss_trajectory.append(epoch_loss / max(1, batches))
    record.epochs = epochs
    record.final_loss = record.loss_trajectory[-1] if record.loss_trajectory else None

    model.eval()
    with torch.no_grad():
        predicted = model.predict_outcomes(features)
        for index, head in enumerate(model.heads):
            record.per_head_final_mse[head] = float(
                torch.nn.functional.mse_loss(predicted[:, index], targets[:, index])
            )
    return record
