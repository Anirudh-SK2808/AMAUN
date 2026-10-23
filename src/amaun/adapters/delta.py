"""M3 -- ``AdapterDelta``: the canonical adapter representation.

Every comparison, feature, fusion, and aggregation in AMAUN operates on **effective
updates** ``Delta_W = scaling * B @ A``, never on the ``A``/``B`` factors separately.

This is a deliberate API constraint, not a style preference.  ``idea.md`` S16.3 forbids
averaging ``A`` and ``B`` independently, because
``mean(B_i) @ mean(A_i) != mean(B_i @ A_i)``.  By making the *delta* the only currency the
rest of the codebase sees, that error becomes hard to commit rather than merely discouraged:
there is no public function that hands out ``A`` and ``B`` for arithmetic.

Deltas are materialised **lazily**.  At 7B with rank 16 over 7 target modules, storing dense
``d_out x d_in`` deltas for every layer would not fit in memory; storing factors costs
``r*(d_in+d_out)``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import torch

from amaun.errors import AdapterError


@dataclass
class LayerDelta:
    """One layer's LoRA factors and scaling.

    ``A: [r, d_in]``, ``B: [d_out, r]``, so ``delta = scaling * B @ A`` has shape
    ``[d_out, d_in]`` -- the same orientation as ``nn.Linear.weight``.
    """

    A: torch.Tensor
    B: torch.Tensor
    scaling: float

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise AdapterError(f"LoRA factors must be 2-D, got A={self.A.shape} B={self.B.shape}")
        if self.A.shape[0] != self.B.shape[1]:
            raise AdapterError(
                f"rank mismatch: A is [r={self.A.shape[0]}, d_in={self.A.shape[1]}] but "
                f"B is [d_out={self.B.shape[0]}, r={self.B.shape[1]}]"
            )

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the *effective* update."""
        return (int(self.B.shape[0]), int(self.A.shape[1]))

    @property
    def numel(self) -> int:
        """Stored parameter count (factors, not the dense delta)."""
        return int(self.A.numel() + self.B.numel())

    def dense(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Materialise ``scaling * B @ A``.

        Only call this on small tensors or when the storage mode explicitly requires a
        dense delta (D-013); it is ``O(d_out * d_in)`` memory.
        """
        target = dtype or torch.float32
        return self.scaling * (self.B.to(target) @ self.A.to(target))

    def apply(self, x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """Compute ``weight * x @ delta.T`` without materialising the delta.

        ``x @ (s*B@A).T == s * (x @ A.T) @ B.T``.  This identity is what makes fusion
        tractable at 7B, and it is exact -- not an approximation.
        """
        inner = torch.nn.functional.linear(x.to(self.A.dtype), self.A)
        return (weight * self.scaling) * torch.nn.functional.linear(inner, self.B)

    def to(self, *args, **kwargs) -> "LayerDelta":
        return LayerDelta(A=self.A.to(*args, **kwargs), B=self.B.to(*args, **kwargs), scaling=self.scaling)

    def detach_clone(self) -> "LayerDelta":
        return LayerDelta(A=self.A.detach().clone(), B=self.B.detach().clone(), scaling=self.scaling)


@dataclass
class AdapterDelta:
    """A full adapter as a mapping ``layer_key -> LayerDelta``.

    ``layer_key`` is the *base module path* the delta applies to (e.g.
    ``model.layers.0.self_attn.q_proj``), normalised away from PEFT's wrapper naming so
    deltas from different sources are comparable by key.
    """

    layers: dict[str, LayerDelta] = field(default_factory=dict)
    adapter_id: str = ""
    #: How the adapter is *stored* (D-013): ``factors`` (LoRA factors, the default) or ``dense``
    #: (``fusion.storage: dense_delta`` keeps one ``d_out x d_in`` fp32 matrix per layer).  The
    #: in-memory form is always exact factors; only the byte accounting differs.
    storage: str = "factors"

    # -- mapping surface ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.layers)

    def __contains__(self, key: object) -> bool:
        return key in self.layers

    def __getitem__(self, key: str) -> LayerDelta:
        return self.layers[key]

    def keys(self) -> list[str]:
        """Layer keys in sorted order.

        Sorted, always: feature vectors and fusion coefficient tensors are indexed by
        layer position, so a nondeterministic key order would silently misalign them.
        """
        return sorted(self.layers)

    def items(self) -> Iterator[tuple[str, LayerDelta]]:
        for key in self.keys():
            yield key, self.layers[key]

    # -- aggregate properties ----------------------------------------------------
    @property
    def rank(self) -> int | None:
        ranks = {layer.rank for layer in self.layers.values()}
        return ranks.pop() if len(ranks) == 1 else None

    @property
    def parameter_count(self) -> int:
        return sum(layer.numel for layer in self.layers.values())

    def storage_bytes(self) -> int:
        if self.storage == "dense":
            return sum(layer.shape[0] * layer.shape[1] * 4 for layer in self.layers.values())
        return sum(
            layer.A.numel() * layer.A.element_size() + layer.B.numel() * layer.B.element_size()
            for layer in self.layers.values()
        )

    def content_hash(self) -> str:
        """Hash of the full tensor content.

        Used to prove a source adapter was *not* mutated by fusion (invariant I8.4) and to
        verify save/load round-trips (I3.1).  Unlike the backbone fingerprint this hashes
        everything, because adapters are small.
        """
        digest = hashlib.sha256()
        for key, layer in self.items():
            digest.update(key.encode())
            digest.update(f"{layer.scaling:.12g}".encode())
            for tensor in (layer.A, layer.B):
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
        return digest.hexdigest()

    def to(self, *args, **kwargs) -> "AdapterDelta":
        return AdapterDelta(
            layers={k: v.to(*args, **kwargs) for k, v in self.layers.items()},
            adapter_id=self.adapter_id, storage=self.storage,
        )

    def detach_clone(self) -> "AdapterDelta":
        return AdapterDelta(
            layers={k: v.detach_clone() for k, v in self.layers.items()},
            adapter_id=self.adapter_id, storage=self.storage,
        )

    # -- persistence -------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save factors + scalings.  Round-trips bit-exactly (invariant I3.1)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapter_id": self.adapter_id,
            "storage": self.storage,
            "layer_keys": self.keys(),
            "scalings": {k: v.scaling for k, v in self.items()},
            "tensors": {f"{k}::{name}": getattr(v, name) for k, v in self.items() for name in ("A", "B")},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AdapterDelta":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        layers: dict[str, LayerDelta] = {}
        for key in payload["layer_keys"]:
            layers[key] = LayerDelta(
                A=payload["tensors"][f"{key}::A"],
                B=payload["tensors"][f"{key}::B"],
                scaling=float(payload["scalings"][key]),
            )
        return cls(layers=layers, adapter_id=payload.get("adapter_id", ""),
                   storage=payload.get("storage", "factors"))

    # -- structural checks -------------------------------------------------------
    def assert_alignable(self, other: "AdapterDelta") -> list[str]:
        """Return the shared layer keys, raising if the shapes are incompatible.

        Two adapters need not cover identical key sets (a rank- or target-module-different
        adapter is a legitimate *incompatible* peer, which the federated compatibility gate
        handles), but where they share a key the effective-update shapes must match or no
        fusion or comparison is meaningful.
        """
        shared = sorted(set(self.layers) & set(other.layers))
        if not shared:
            raise AdapterError(
                f"adapters '{self.adapter_id}' and '{other.adapter_id}' share no layer keys; "
                f"they cannot be compared or fused"
            )
        for key in shared:
            if self.layers[key].shape != other.layers[key].shape:
                raise AdapterError(
                    f"layer '{key}' effective-update shape mismatch: "
                    f"{self.layers[key].shape} vs {other.layers[key].shape}"
                )
        return shared

    def summary(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "num_layers": len(self),
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes(),
            "content_hash": self.content_hash(),
        }


def combine_deltas(
    components: list[AdapterDelta],
    weights: Mapping[str, torch.Tensor] | list[float],
) -> AdapterDelta:
    """Fuse deltas at the effective-update level: ``sum_k w_k * Delta_W_k``.

    ``weights`` is either one scalar per component (global coefficients) or a mapping
    ``layer_key -> tensor[m]`` (layer-wise coefficients).  The result is stored *densely*
    as a rank-``sum(r_k)`` factorisation obtained by concatenation -- which is exact:

        ``sum_k w_k * s_k * B_k @ A_k == [w_1 s_1 B_1 | ... | w_m s_m B_m] @ [A_1; ...; A_m]``

    so no SVD approximation is introduced here.  Re-factorising down to a smaller rank is a
    separate, explicitly lossy step (see ``fusion/refactorize.py``, D-013).
    """
    if not components:
        raise AdapterError("cannot combine an empty list of deltas")
    shared = set(components[0].layers)
    for other in components[1:]:
        shared &= set(other.layers)
    if not shared:
        raise AdapterError("components share no layer keys")

    out: dict[str, LayerDelta] = {}
    for key in sorted(shared):
        if isinstance(weights, Mapping):
            layer_weights = weights[key]
        else:
            layer_weights = torch.tensor(list(weights), dtype=torch.float32)
        if len(layer_weights) != len(components):
            raise AdapterError(
                f"expected {len(components)} coefficients for layer '{key}', got {len(layer_weights)}"
            )
        a_blocks, b_blocks = [], []
        for weight, component in zip(layer_weights, components):
            layer = component.layers[key]
            a_blocks.append(layer.A.to(torch.float32))
            b_blocks.append(float(weight) * layer.scaling * layer.B.to(torch.float32))
        out[key] = LayerDelta(
            A=torch.cat(a_blocks, dim=0),
            B=torch.cat(b_blocks, dim=1),
            scaling=1.0,  # scalings were folded into B above
        )
    return AdapterDelta(layers=out, adapter_id="+".join(c.adapter_id for c in components))
