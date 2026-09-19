"""M6.7 -- resource features (idea.md S4.6).

    Mem_i     = |phi_i| * b            measured from real tensors, not a formula
    Comm_i    = Mem_i + |meta_i|       measured from real serialised bytes
    RepoCost  = sum_j Mem_j
    Latency_i = T_infer(A_i)           measured under a fixed profiling protocol

Everything here is *measured*.  An estimated memory or communication figure would be a
fabricated result under master-prompt constraint 1, and these numbers feed directly into the
efficiency claims the paper would make.
"""

from __future__ import annotations

import json
from typing import Any

from amaun.features.base import FeatureFamily


class ResourceFeatures(FeatureFamily):
    name = "resource"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        delta = ctx.candidate_delta
        metadata = ctx.candidate_metadata
        memory_bytes = float(delta.storage_bytes())
        metadata_bytes = float(len(json.dumps(metadata.as_dict(), default=str).encode()))
        repository_bytes = float(getattr(ctx, "repository_bytes", 0.0) or 0.0)

        return {
            "Mem_i": memory_bytes,
            "Comm_i": memory_bytes + metadata_bytes,
            "RepoCost": repository_bytes,
            "MemShare_i": memory_bytes / repository_bytes if repository_bytes > 0 else None,
            "Params_i": float(delta.parameter_count),
            "Latency_i": self.finite(getattr(ctx, "candidate_latency_seconds", None)),
            "RepoSize": float(getattr(ctx, "repository_size", 0) or 0),
        }
