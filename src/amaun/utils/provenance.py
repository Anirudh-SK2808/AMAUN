"""Provenance capture: everything needed to reproduce a run (brief S18)."""

from __future__ import annotations

import hashlib
import importlib
import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any

_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "numpy",
    "scipy",
    "accelerate",
    "safetensors",
    "bitsandbytes",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def git_state() -> dict[str, Any]:
    """Git provenance, including a hash that pins *uncommitted* state.

    ``dirty`` alone cannot prove that two dirty runs had identical working trees, so we hash
    the working state.  That hash must cover **untracked** files too: in a fresh repository
    (or before the first commit) ``git diff HEAD`` is empty while the tree is unmistakably
    dirty, and a null hash there would silently lose the only record of what actually ran.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD") if commit else None
    dirty = bool(status)

    working_state: str | None = None
    if dirty or diff:
        # Combine the tracked diff with the porcelain status (which lists untracked paths),
        # so the hash is defined whenever the tree is dirty for any reason.
        working_state = "\n".join(filter(None, [diff or "", status or ""]))

    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "has_commits": bool(commit),
        "dirty": dirty,
        "dirty_files": [line[3:] for line in (status or "").splitlines()],
        "diff_sha256": (
            hashlib.sha256(working_state.encode()).hexdigest() if working_state is not None else None
        ),
        "diff_covers_untracked": True,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "cpu_count": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        pass
    return info


def collect_provenance() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware(),
    }
