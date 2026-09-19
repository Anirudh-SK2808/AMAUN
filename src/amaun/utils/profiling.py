"""Measured (never estimated) cost accounting.

idea.md S8 asks for a cost model implemented as *profiling utilities*.  Every number here
is measured from the live process, because an estimated communication or memory figure
would be a fabricated result under master_prompt constraint 1.
"""

from __future__ import annotations

import io
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Timing:
    label: str
    seconds: float


@contextmanager
def timed(label: str, sink: list[Timing] | None = None) -> Iterator[Timing]:
    start = time.perf_counter()
    record = Timing(label=label, seconds=0.0)
    try:
        yield record
    finally:
        record.seconds = time.perf_counter() - start
        if sink is not None:
            sink.append(record)


def peak_memory_bytes(device: str) -> int | None:
    """Peak allocator memory, where the backend can report it."""
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
        if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
    except Exception:  # pragma: no cover
        return None
    return None


def reset_peak_memory(device: str) -> None:
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass


def serialized_size_bytes(payload: Any) -> int:
    """Real serialised byte count of a tensor payload.

    Used for federated communication accounting (invariant I11.4).  We serialise with
    ``torch.save`` into memory rather than summing ``numel * itemsize`` so that metadata
    and dtype overhead are included -- the payload that would actually travel.
    """
    import torch

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getbuffer().nbytes


def count_parameters(module: Any) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


@dataclass
class CostLedger:
    """Accumulates measured costs for one run, mirroring idea.md S8."""

    timings: list[Timing] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, value: float) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + float(value)

    @contextmanager
    def time(self, label: str) -> Iterator[Timing]:
        with timed(label, self.timings) as record:
            yield record
        self.add(f"seconds.{label}", record.seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timings": [{"label": t.label, "seconds": t.seconds} for t in self.timings],
            "counters": dict(self.counters),
        }
