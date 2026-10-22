"""The REUSE gate -- stage 1 of ``manager.decision_order: prescreen_then_train`` (D-003 O1).

Decides REUSE *before* any temporary adapter is trained, so a reused task costs no training:

    task arrives
      -> stage 1  REUSE gate on temp-free evidence (SUPPORT only)
           fires -> REUSE the best existing adapter; nothing is trained
           else  -> stage 2: train A_temp -> full utility network -> FUSE | CREATE

**Evidence.** Only features that exist without ``A_temp``:

- ``d_task_sq``: squared Euclidean distance between task embeddings (as ``semantic.d_task_sq``);
- the candidate's loss / confidence / entropy / ECE / token accuracy on the new task's SUPPORT;
- the frozen backbone's loss on the same batches, and the candidate's gain over it.

Parameter, gradient and adapter-vs-adapter divergence features all need ``A_temp`` and are
therefore not available here; the full utility network still uses them in stage 2.

**Prediction.** The gate predicts the *reuse gap*

    gap = score(A_cand on new task) - score(dedicated adapter on new task)

measured on VALIDATION with the supervision proxy ``exp(-loss / loss_to_score_scale)``.  It is a
ridge regression over the standardised features, so no single metric is the rule (idea.md
S16.2).  The gate fires when the best predicted gap is ``>= -tolerance`` and that candidate
beats the frozen backbone on SUPPORT.  ``tolerance`` is a definition ("how much worse than a
dedicated adapter is still acceptable"), not a value fitted to make the gate fire.

**Supervision.** Fitted once on decision-development measurements (VALIDATION labels, SUPPORT
features), then frozen.  Normalisation statistics come from that split only (idea.md S4.7).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amaun.data.preprocess import collate
from amaun.data.registry import TaskSpec
from amaun.data.splits import SplitName, TaskSplits, manager_guard
from amaun.errors import ManagerError
from amaun.utils.logging import get_logger

log = get_logger("manager.reuse_gate")

GATE_FEATURES = (
    "d_task_sq",
    "cand_loss",
    "base_loss",
    "loss_gain",
    "rel_loss_gain",
    "cand_confidence",
    "cand_entropy",
    "cand_ece",
    "cand_token_accuracy",
)


@dataclass(frozen=True)
class ReuseGateConfig:
    enabled: bool
    tolerance: float
    ridge_lambda: float
    probe_examples: int
    require_gain_over_backbone: bool
    min_train_examples: int

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "ReuseGateConfig":
        required = ("enabled", "tolerance", "ridge_lambda", "probe_examples",
                    "require_gain_over_backbone", "min_train_examples")
        missing = [key for key in required if key not in section]
        if missing:
            raise ManagerError(f"utility.reuse_gate missing required keys: {missing}")
        return cls(
            enabled=bool(section["enabled"]),
            tolerance=float(section["tolerance"]),
            ridge_lambda=float(section["ridge_lambda"]),
            probe_examples=int(section["probe_examples"]),
            require_gain_over_backbone=bool(section["require_gain_over_backbone"]),
            min_train_examples=int(section["min_train_examples"]),
        )


# --------------------------------------------------------------------------------------
# Features -- one code path for supervision and for decisions
# --------------------------------------------------------------------------------------


def support_probe_batches(tokenizer: Any, task: TaskSpec, splits: TaskSplits, *, count: int) -> list:
    """SUPPORT batches (pairs, as the runner's probes) -- the only split the gate reads."""
    examples = list(manager_guard(splits).get(SplitName.SUPPORT))[:count]
    return [
        collate(tokenizer, examples[i : i + 2], max_input_tokens=task.max_input_tokens,
                max_target_tokens=task.max_target_tokens, prompt_truncation=task.prompt_truncation)
        for i in range(0, len(examples), 2)
    ]


def _squared_distance(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None:
        return None
    a = torch.as_tensor(left, dtype=torch.float32).flatten()
    b = torch.as_tensor(right, dtype=torch.float32).flatten()
    if a.numel() == 0 or a.numel() != b.numel():
        return None
    return float(torch.sum((a - b) ** 2))


def compute_gate_features(
    *,
    model: Any,
    device: str,
    probe_batches: Sequence[Any],
    task_embedding: Sequence[float] | None,
    candidates: Sequence[Any],
    ece_bins: int,
) -> dict[str, dict[str, float | None]]:
    """Gate features for every candidate ``AdapterEntry`` on one task's SUPPORT probe batches."""
    from amaun.features.extractor import probe_quality

    if not probe_batches:
        raise ManagerError("the REUSE gate needs SUPPORT probe batches; the task has none")
    base = probe_quality(model, None, probe_batches, device, ece_bins=ece_bins)
    out: dict[str, dict[str, float | None]] = {}
    for entry in candidates:
        cand = probe_quality(model, entry.delta, probe_batches, device, ece_bins=ece_bins)
        gain = base["loss"] - cand["loss"]
        out[entry.adapter_id] = {
            "d_task_sq": _squared_distance(task_embedding, entry.metadata.task_embedding),
            "cand_loss": cand["loss"],
            "base_loss": base["loss"],
            "loss_gain": gain,
            "rel_loss_gain": gain / base["loss"] if base["loss"] else None,
            "cand_confidence": cand["confidence"],
            "cand_entropy": cand["entropy"],
            "cand_ece": cand["ece"],
            "cand_token_accuracy": cand["token_accuracy"],
        }
    return out


def reuse_gap(candidate_val_loss: float, dedicated_val_loss: float, *, scale: float) -> float:
    """Measured gap in the supervision proxy: ``exp(-L_cand/s) - exp(-L_dedicated/s)``."""
    def score(loss: float) -> float:
        return math.exp(-max(0.0, loss) / scale) if math.isfinite(loss) and scale > 0 else 0.0
    return score(candidate_val_loss) - score(dedicated_val_loss)


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


@dataclass
class GateExample:
    task_id: str
    candidate_id: str
    features: dict[str, float | None]
    gap: float                      # measured on VALIDATION


@dataclass
class GateDecision:
    fired: bool
    adapter_id: str | None
    predicted_gaps: dict[str, float]
    eligible: dict[str, bool]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReuseGate:
    config: ReuseGateConfig
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    n_train: int
    run_mode: str
    trained: bool
    notes: list[str] = field(default_factory=list)
    feature_names: tuple[str, ...] = GATE_FEATURES

    @staticmethod
    def _matrix(rows: Sequence[Mapping[str, float | None]], fill: Sequence[float]) -> torch.Tensor:
        return torch.tensor(
            [[float(row.get(name)) if row.get(name) is not None and math.isfinite(float(row.get(name)))
              else fill[i] for i, name in enumerate(GATE_FEATURES)] for row in rows],
            dtype=torch.float64,
        )

    @classmethod
    def fit(cls, examples: Sequence[GateExample], config: ReuseGateConfig, *, run_mode: str) -> "ReuseGate":
        """Closed-form ridge regression of the measured gap on standardised gate features."""
        notes: list[str] = []
        dim = len(GATE_FEATURES)
        if len(examples) < max(2, config.min_train_examples):
            notes.append(
                f"only {len(examples)} gate example(s) (< min_train_examples="
                f"{config.min_train_examples}); the gate is DISABLED and never fires"
            )
            return cls(config, [0.0] * dim, [1.0] * dim, [0.0] * dim, 0.0, len(examples),
                       run_mode, trained=False, notes=notes)

        # Column means over observed values only; missing values are imputed with them, so a
        # missing feature contributes nothing after standardisation.
        means = []
        for name in GATE_FEATURES:
            observed = [float(e.features[name]) for e in examples
                        if e.features.get(name) is not None and math.isfinite(float(e.features[name]))]
            means.append(sum(observed) / len(observed) if observed else 0.0)
            if not observed:
                notes.append(f"feature '{name}' missing on every gate example")
        x = cls._matrix([e.features for e in examples], means)
        mu = x.mean(dim=0)
        sd = x.std(dim=0, unbiased=False)
        sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
        z = (x - mu) / sd
        y = torch.tensor([e.gap for e in examples], dtype=torch.float64)
        y_mean = y.mean()
        w = torch.linalg.solve(z.T @ z + config.ridge_lambda * torch.eye(dim, dtype=torch.float64),
                               z.T @ (y - y_mean))
        gaps = [e.gap for e in examples]
        if all(g < -config.tolerance for g in gaps):
            notes.append("every measured gap was below -tolerance: on this evidence reuse was never "
                         "acceptable, so the gate is expected to stay closed")
        elif all(g >= -config.tolerance for g in gaps):
            notes.append("every measured gap was within tolerance: no negative examples were observed")
        log.info("reuse gate fitted on %d examples (ridge lambda=%.3g)", len(examples), config.ridge_lambda)
        return cls(config, mu.tolist(), sd.tolist(), w.tolist(), float(y_mean), len(examples),
                   run_mode, trained=True, notes=notes)

    def predict(self, features: Mapping[str, float | None]) -> float:
        x = self._matrix([features], self.mean)[0]
        z = (x - torch.tensor(self.mean, dtype=torch.float64)) / torch.tensor(self.std, dtype=torch.float64)
        return float(z @ torch.tensor(self.weights, dtype=torch.float64) + self.bias)

    def decide(self, candidate_features: Mapping[str, Mapping[str, float | None]], *, run_mode: str) -> GateDecision:
        if run_mode != self.run_mode:
            raise ManagerError(
                f"reuse gate was fitted under run_mode '{self.run_mode}' but is applied under "
                f"'{run_mode}'; gate statistics must never cross run modes (idea.md S4.7)"
            )
        if not self.config.enabled or not self.trained:
            return GateDecision(False, None, {}, {}, "gate disabled or untrained: proceed to training")
        predicted = {cid: self.predict(f) for cid, f in candidate_features.items()}
        eligible = {
            cid: (not self.config.require_gain_over_backbone)
            or (f.get("loss_gain") is not None and float(f["loss_gain"]) > 0.0)
            for cid, f in candidate_features.items()
        }
        pool = [cid for cid in predicted if eligible[cid]]
        if not pool:
            return GateDecision(False, None, predicted, eligible,
                                "no candidate improves on the frozen backbone on SUPPORT: train")
        best = max(pool, key=lambda cid: (predicted[cid], cid))
        if predicted[best] >= -self.config.tolerance:
            return GateDecision(True, best, predicted, eligible,
                                f"predicted reuse gap {predicted[best]:.4f} >= -{self.config.tolerance}: "
                                f"reuse '{best}' without training")
        return GateDecision(False, None, predicted, eligible,
                            f"best predicted reuse gap {predicted[best]:.4f} < -{self.config.tolerance}: train")

    # -- persistence -------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config), "feature_names": list(self.feature_names),
            "mean": self.mean, "std": self.std, "weights": self.weights, "bias": self.bias,
            "n_train": self.n_train, "run_mode": self.run_mode, "trained": self.trained,
            "notes": list(self.notes),
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str, *, expected_run_mode: str) -> "ReuseGate":
        data = json.loads(Path(path).read_text())
        if data["run_mode"] != expected_run_mode:
            raise ManagerError(f"reuse gate at {path} is for run_mode '{data['run_mode']}', "
                               f"not '{expected_run_mode}'")
        if tuple(data["feature_names"]) != GATE_FEATURES:
            raise ManagerError(f"reuse gate at {path} has a different feature layout")
        return cls(ReuseGateConfig(**data["config"]), data["mean"], data["std"], data["weights"],
                   float(data["bias"]), int(data["n_train"]), data["run_mode"], bool(data["trained"]),
                   list(data.get("notes", [])))


def gate_reuse_decision(
    gate: GateDecision,
    *,
    task_id: str,
    task_index: int,
    thresholds: Any,
    repository_size: int,
    config_hash: str,
) -> Any:
    """The Decision record for a REUSE emitted by the gate: no temporary adapter was trained."""
    import uuid

    from amaun.manager.decision import CandidateScore, Decision, Outcome

    ranked = sorted(
        (CandidateScore(adapter_id=cid, utility=gap, outcomes={"predicted_reuse_gap": gap},
                        feature_signature="reuse_gate:" + "+".join(GATE_FEATURES),
                        n_features=len(GATE_FEATURES))
         for cid, gap in gate.predicted_gaps.items()),
        key=lambda s: s.utility, reverse=True,
    )
    return Decision(
        decision_id=f"d-{task_index:04d}-{uuid.uuid4().hex[:8]}",
        outcome=Outcome.REUSE,
        task_id=task_id,
        task_index=task_index,
        selected_adapter_ids=[gate.adapter_id],
        ranked_candidates=ranked,
        tau_u=getattr(thresholds, "tau_u", None),
        tau_reuse=getattr(thresholds, "tau_reuse", None),
        threshold_source="reuse_gate",
        decision_order="prescreen_then_train",
        reason="REUSE gate (before training): " + gate.reason,
        temp_adapter_trained=False,
        config_hash=config_hash,
        repository_size_before=repository_size,
        n_candidates_scored=len(ranked),
    )
