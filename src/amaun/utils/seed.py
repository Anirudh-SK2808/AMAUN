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
            torch.use_deterministic_algorithms(True, warn_only=True)
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
