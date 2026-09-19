"""M11 -- communication accounting (idea.md S8, invariant I11.4).

    Comm_full = |theta| * b        conceptual full-model payload (NEVER actually sent)
    Comm_LoRA = sum_l |phi_l| * b + |meta|

Payload sizes are **measured** by serialising the real payload, not estimated from parameter
counts, so dtype and container overhead are included.  An estimated communication figure
would be a fabricated result, and communication cost is one of the efficiency claims.

:func:`assert_no_backbone` inspects the payload for tensors large enough to be backbone
weights and for backbone-shaped key names, enforcing master-prompt constraint 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.metadata import AdapterMetadata
from amaun.errors import PayloadContainsBackboneError
from amaun.utils.profiling import serialized_size_bytes


@dataclass
class Payload:
    """What a client actually transmits: LoRA factors plus lightweight metadata."""

    adapter_id: str
    client_id: str
    tensors: dict[str, Any]
    metadata: dict[str, Any]

    def size_bytes(self) -> int:
        tensor_bytes = serialized_size_bytes(self.tensors)
        metadata_bytes = len(json.dumps(self.metadata, default=str).encode())
        return tensor_bytes + metadata_bytes


@dataclass
class CommunicationLedger:
    """Cumulative, measured communication per round."""

    per_round: dict[int, int] = field(default_factory=dict)
    per_client: dict[str, int] = field(default_factory=dict)
    conceptual_full_model_bytes: int | None = None
    n_transmissions: int = 0

    def record(self, round_index: int, client_id: str, payload_bytes: int) -> None:
        self.per_round[round_index] = self.per_round.get(round_index, 0) + payload_bytes
        self.per_client[client_id] = self.per_client.get(client_id, 0) + payload_bytes
        self.n_transmissions += 1

    @property
    def total_bytes(self) -> int:
        return sum(self.per_round.values())

    def counterfactual_full_model_bytes(self) -> int | None:
        """What the same protocol would have cost sending full models instead of adapters.

        The comparison must be like-for-like: ``n_transmissions * |theta| * b``.  Comparing
        *cumulative* LoRA traffic against a *single* full-model payload would understate the
        saving by a factor of ``n_transmissions`` and is not a meaningful figure.
        """
        if not self.conceptual_full_model_bytes:
            return None
        return self.n_transmissions * self.conceptual_full_model_bytes

    def savings_vs_full_model(self) -> float | None:
        """Ratio of actual LoRA traffic to the like-for-like full-model counterfactual."""
        counterfactual = self.counterfactual_full_model_bytes()
        if not counterfactual:
            return None
        return self.total_bytes / counterfactual

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_round_bytes": dict(self.per_round),
            "per_client_bytes": dict(self.per_client),
            "total_bytes": self.total_bytes,
            "n_transmissions": self.n_transmissions,
            "full_model_bytes_per_transmission": self.conceptual_full_model_bytes,
            "counterfactual_full_model_bytes": self.counterfactual_full_model_bytes(),
            "lora_to_full_ratio": self.savings_vs_full_model(),
        }


# Any single tensor above this many elements is far larger than a LoRA factor at realistic
# ranks and indicates a backbone weight slipped into the payload.
_SUSPICIOUS_TENSOR_ELEMENTS = 50_000_000
_BACKBONE_NAME_MARKERS = ("embed_tokens", "lm_head", "layernorm", "norm.weight", "base_model.weight")


def assert_no_backbone(payload: Payload) -> None:
    """Master-prompt constraint 4: the backbone is never transmitted."""
    for key, tensor in payload.tensors.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _BACKBONE_NAME_MARKERS):
            raise PayloadContainsBackboneError(
                f"payload key '{key}' looks like a backbone tensor; the frozen backbone is "
                f"shared and must never be transmitted (idea.md S7.3)"
            )
        numel = getattr(tensor, "numel", None)
        if callable(numel) and numel() > _SUSPICIOUS_TENSOR_ELEMENTS:
            raise PayloadContainsBackboneError(
                f"payload tensor '{key}' has {numel()} elements, far larger than a LoRA factor; "
                f"this is almost certainly a backbone weight"
            )


def build_payload(
    *, client_id: str, delta: AdapterDelta, metadata: AdapterMetadata
) -> Payload:
    """Serialise one client's adapter into a transmittable payload, then verify it."""
    tensors: dict[str, Any] = {}
    for key, layer in delta.items():
        tensors[f"{key}::A"] = layer.A
        tensors[f"{key}::B"] = layer.B
        tensors[f"{key}::scaling"] = layer.scaling
    payload = Payload(
        adapter_id=delta.adapter_id,
        client_id=client_id,
        tensors=tensors,
        metadata={
            "adapter_id": metadata.adapter_id,
            "task_name": metadata.task_name,
            "tasks_served": metadata.tasks_served,
            "lora_rank": metadata.lora_rank,
            "target_modules": metadata.target_modules,
            "backbone_fingerprint": metadata.backbone_fingerprint,
            "origin": metadata.origin,
        },
    )
    assert_no_backbone(payload)
    return payload
