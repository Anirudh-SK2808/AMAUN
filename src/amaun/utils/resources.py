"""Measured resource monitoring: GPU allocator state, CPU RSS, and live-tensor counts.

Distinguishing a leak from allocator caching needs three different numbers, so all three are
recorded at every checkpoint:

- ``allocated`` -- bytes held by live tensors.  This is what grows if a reference is leaked.
- ``reserved``  -- bytes the caching allocator keeps from the driver.  It is allowed to stay
  high after tensors are freed; growth here without growth in ``allocated`` is caching or
  fragmentation, not a leak.
- ``live_tensors`` -- the number of tensor objects reachable from Python (``gc``).  A stable
  ``allocated`` with a rising ``live_tensors`` count means small tensors (CPU metrics,
  gradients, cached logits) are being retained across cycles.

Every value is read from the live process; nothing is estimated (master_prompt constraint 1).
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


def _cpu_rss_bytes() -> int | None:
    """Current resident set size of this process."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:  # Linux fallback without psutil
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def live_tensor_count() -> int:
    """Number of ``torch.Tensor`` objects currently reachable through the garbage collector."""
    import warnings

    import torch

    count = 0
    with warnings.catch_warnings():
        # isinstance() on some lazily-deprecated module attributes emits FutureWarnings.
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor):
                    count += 1
            except Exception:  # some objects raise on isinstance through proxies
                continue
    return count


def memory_snapshot(device: str, *, label: str = "", count_tensors: bool = False) -> dict[str, Any]:
    """One measured snapshot.  ``count_tensors`` walks the GC heap, so it is opt-in."""
    import torch

    snap: dict[str, Any] = {
        "label": label,
        "ts": time.time(),
        "device": device,
        "cpu_rss_bytes": _cpu_rss_bytes(),
    }
    if device == "cuda" and torch.cuda.is_available():
        snap.update(
            gpu_allocated_bytes=int(torch.cuda.memory_allocated()),
            gpu_reserved_bytes=int(torch.cuda.memory_reserved()),
            gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            snap.update(
                gpu_allocated_bytes=int(torch.mps.current_allocated_memory()),
                gpu_reserved_bytes=int(torch.mps.driver_allocated_memory()),
            )
        except Exception:
            pass
    if count_tensors:
        snap["live_tensors"] = live_tensor_count()
    return snap


@dataclass
class ResourceMonitor:
    """Appends labelled snapshots to a JSONL file and keeps them for the stability report."""

    device: str
    path: Path | None = None
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def record(self, label: str, *, cycle: int | None = None, count_tensors: bool = False,
               **extra: Any) -> dict[str, Any]:
        snap = memory_snapshot(self.device, label=label, count_tensors=count_tensors)
        if cycle is not None:
            snap["cycle"] = cycle
        snap.update(extra)
        self.snapshots.append(snap)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(snap) + "\n")
        return snap


def cleanup(device: str, *, release_cache: bool = False) -> None:
    """Collect garbage; optionally return cached blocks to the driver.

    ``release_cache`` is only for measuring (so ``reserved`` reflects live use) or before a
    large allocation in a different phase.  It does not fix a leak: a retained reference keeps
    its memory regardless of the cache.
    """
    import torch

    gc.collect()
    if release_cache:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()


def stability_report(
    cycle_snapshots: Sequence[dict[str, Any]],
    *,
    allocated_tolerance_bytes: int,
    tensor_tolerance: int,
) -> dict[str, Any]:
    """Judge whether post-cleanup memory returns to a stable baseline across repeated cycles.

    ``cycle_snapshots`` are the *post-cleanup* snapshots of each cycle, in order.  The first
    cycle is the warm-up (allocator pools, lazily created kernels, cached tokenizer state), so
    the baseline is cycle 1 and growth is measured from cycle 1 to the last cycle.
    """
    if len(cycle_snapshots) < 3:
        return {
            "verdict": "insufficient_cycles",
            "detail": "need >= 3 cycles: one warm-up plus two to compare",
            "n_cycles": len(cycle_snapshots),
        }
    baseline, last = cycle_snapshots[1], cycle_snapshots[-1]

    def delta(key: str) -> int | None:
        if baseline.get(key) is None or last.get(key) is None:
            return None
        return int(last[key]) - int(baseline[key])

    allocated_growth = delta("gpu_allocated_bytes")
    reserved_growth = delta("gpu_reserved_bytes")
    tensor_growth = delta("live_tensors")
    rss_growth = delta("cpu_rss_bytes")

    problems = []
    if allocated_growth is not None and allocated_growth > allocated_tolerance_bytes:
        problems.append(f"GPU allocated grew {allocated_growth} B after warm-up")
    if tensor_growth is not None and tensor_growth > tensor_tolerance:
        problems.append(f"live tensor count grew by {tensor_growth} after warm-up")
    verdict = "leak_suspected" if problems else "stable"
    notes = []
    if verdict == "stable" and reserved_growth and reserved_growth > allocated_tolerance_bytes:
        notes.append(
            "reserved memory grew while allocated memory did not: allocator caching or "
            "fragmentation, not a leaked reference"
        )
    return {
        "verdict": verdict,
        "problems": problems,
        "notes": notes,
        "n_cycles": len(cycle_snapshots),
        "baseline_cycle": 1,
        "gpu_allocated_growth_bytes": allocated_growth,
        "gpu_reserved_growth_bytes": reserved_growth,
        "live_tensor_growth": tensor_growth,
        "cpu_rss_growth_bytes": rss_growth,
        "allocated_tolerance_bytes": allocated_tolerance_bytes,
        "tensor_tolerance": tensor_tolerance,
        "per_cycle": [
            {k: s.get(k) for k in ("cycle", "gpu_allocated_bytes", "gpu_reserved_bytes",
                                   "live_tensors", "cpu_rss_bytes")}
            for s in cycle_snapshots
        ],
    }
