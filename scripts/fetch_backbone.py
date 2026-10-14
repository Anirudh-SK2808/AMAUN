#!/usr/bin/env python
"""Download, pin, and offline-verify the run mode's backbone checkpoint.

PHASE 0 of the execution order: nothing else in a ``local_smoke`` or ``actual`` run can be
trusted until the exact checkpoint the run will use exists locally and its commit sha is
recorded (D-001, FR-1.3).

The script is deliberately narrow:

- it refuses ``--run-mode dummy``, whose backbone is built in-process and has no checkpoint;
- it downloads into ``model.cache_dir`` so a second run needs no network;
- it raises if the resolved sha differs from a pinned :attr:`BackboneSpec.revision`, because a
  silently different checkpoint changes every adapter trained against it;
- it then re-loads with ``local_files_only=True`` and records whether that succeeded, so
  "offline-reusable" is a measured property rather than an assumption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.config import RunMode
from amaun.errors import ConfigError
from amaun.experiment import add_common_arguments, resolve_from_args
from amaun.models.backbone import BackboneCache, load_backbone
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    # Refused *before* resolve()/any network call: the dummy tier has no checkpoint at all.
    if args.run_mode == RunMode.DUMMY.value:
        raise ConfigError(
            "run_mode=dummy builds a tiny randomly-initialised Qwen2 in-process and has no "
            "upstream checkpoint to download (BACKBONE_REGISTRY[DUMMY].model_id is None). "
            "Use --run-mode local_smoke or actual."
        )

    cfg = resolve_from_args(args)
    spec = cfg.backbone
    if spec.model_id is None:
        raise ConfigError(f"run_mode={cfg.run_mode.value} has no model_id to fetch")

    cache_dir = cfg.get("model.cache_dir")          # mandatory: no code default for a cache root
    pinned = spec.revision                           # may be None on a first fetch
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, snapshot_download

    resolved_sha = HfApi().model_info(spec.model_id, revision=pinned or "main").sha
    if pinned is not None and pinned != resolved_sha:
        raise ConfigError(
            f"BACKBONE_REGISTRY[{cfg.run_mode.value}].revision is pinned to {pinned} but "
            f"{spec.model_id} resolves that revision to {resolved_sha}. Refusing to continue: a "
            f"different checkpoint would change every adapter trained against it (FR-1.4)."
        )

    local_path = snapshot_download(
        repo_id=spec.model_id,
        revision=resolved_sha,
        cache_dir=cache_dir,
    )
    files = sorted(
        str(path.relative_to(local_path))
        for path in Path(local_path).rglob("*")
        if path.is_file()
    )
    bytes_total = sum(
        (Path(local_path) / name).stat().st_size for name in files
    )

    # Offline re-load verification.  Release the single-slot cache on both sides so no second
    # backbone becomes resident (invariant I1.4).
    BackboneCache.release()
    offline_cfg = cfg
    verified_offline_load = False
    offline_error: str | None = None
    if pinned == resolved_sha:
        tree = dict(cfg.raw)
        tree["model"] = {**tree.get("model", {}), "local_files_only": True}
        from amaun.config import resolve as resolve_config

        offline_cfg = resolve_config(
            tree, run_mode=cfg.run_mode, experiment_id=cfg.experiment_id,
            seed=cfg.seed, device=cfg.device,
        )
        try:
            loaded = load_backbone(offline_cfg)
            verified_offline_load = bool(loaded.frozen_parameter_count > 0)
        except Exception as exc:  # recorded, never masked
            offline_error = f"{type(exc).__name__}: {exc}"
        finally:
            BackboneCache.release()
    else:
        offline_error = (
            "skipped: BACKBONE_REGISTRY has no pinned revision yet, so an offline load would "
            "resolve 'main' over the network. Record the sha below in BACKBONE_REGISTRY and "
            "re-run to verify (T-04)."
        )

    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    record = {
        "repo_id": spec.model_id,
        "run_mode": cfg.run_mode.value,
        "revision": resolved_sha,
        "registry_revision": pinned,
        "registry_revision_matches": pinned == resolved_sha,
        "cache_dir": str(cache_dir),
        "snapshot_path": str(local_path),
        "files": files,
        "bytes_total": bytes_total,
        "verified_offline_load": verified_offline_load,
        "offline_load_error": offline_error,
        "label": smoke_label(cfg.run_mode),
    }
    path = tracker.write_json("backbone_cache.json", record)

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"repo_id                {spec.model_id}")
    print(f"resolved revision      {resolved_sha}")
    print(f"registry revision      {pinned} (matches={pinned == resolved_sha})")
    print(f"snapshot               {local_path}")
    print(f"files / bytes          {len(files)} / {bytes_total}")
    print(f"verified offline load  {verified_offline_load}")
    if offline_error:
        print(f"offline load note      {offline_error}")
    print(f"record                 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
