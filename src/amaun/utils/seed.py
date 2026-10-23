"""Deterministic seeding.

The independence guarantee for the six initial adapters (brief S1) requires that training
adapter *k* does not depend on whether adapters ``1..k-1`` were trained first in the same
process.  We achieve that by deriving a per-adapter seed from the base seed and the adapter
id, and re-seeding immediately before each adapter's training run.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

#: Set by :func:`configure_determinism`.  When True, every later ``set_all_seeds`` keeps strict
#: deterministic algorithms instead of re-enabling the warn-only mode.
_STRICT_DETERMINISM = False


def derive_seed(base_seed: int, key: str) -> int:
    """Stable 31-bit seed from ``(base_seed, key)``.

    Uses sha256 rather than ``hash()`` because Python's string hashing is salted per
    process, which would break reproducibility across runs.
    """
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def set_all_seeds(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every RNG AMAUN touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # warn_only: several ops (e.g. some MPS/SDPA kernels) have no deterministic
            # implementation; we prefer a recorded warning over a hard crash, and the
            # provenance record notes that determinism was best-effort.
            torch.use_deterministic_algorithms(True, warn_only=not _STRICT_DETERMINISM)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function (torch passes the worker id)."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:  # pragma: no cover
        pass


def configure_determinism(enabled: bool, *, warn_only: bool = False) -> dict[str, Any]:
    """``deterministic: true`` (l4_full_study item 11).  Call before the backbone is loaded.

    Sets ``CUBLAS_WORKSPACE_CONFIG`` (cuBLAS needs a fixed workspace to be deterministic, and reads
    it when the CUDA context is created), strict ``torch.use_deterministic_algorithms`` -- an op
    with no deterministic kernel then raises, which the preflight's train / fusion / generation
    steps surface before any long phase -- and cuDNN's deterministic mode.  Data loaders are
    already seeded (``BatchPlan.seed``).  Returns what was set, for the run record.
    """
    global _STRICT_DETERMINISM
    if not enabled:
        return {"deterministic": False}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    _STRICT_DETERMINISM = not warn_only
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"deterministic": True, "warn_only": warn_only,
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
