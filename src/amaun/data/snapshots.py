"""M2 -- a content-keyed, bounded, offline-reusable dataset cache (FR-5).

Streaming from the Hub never populates a local cache, so without this module every run is as
network-dependent as the first and two runs can silently see different rows.  The cache stores
the *capped sample* that was actually consumed -- a few MB -- never a full corpus (brief S4).

Two properties carry the weight:

1. **The key hashes every content-determining input** (dataset id, revision, config, upstream
   split, requested size, ``skip``, sampling policy + buffer size, row adapter + its options,
   and the adapter's required field list).  Change any one of them and you get a different
   file: a stale snapshot can never be silently reused under new coordinates.
2. **Raw upstream rows are stored, not rendered examples.**  A prompt-template edit is a cheap
   local formatting change and must not force a re-download.  The cost is that the file cannot
   prove which prompts were produced -- which is fine, because ``split_provenance`` and
   ``config_used.json`` already record the template-bearing ``TaskSpec``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from amaun.errors import DataError

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SnapshotSpec:
    """Everything that determines *which rows* a load produces."""

    hf_id: str
    hf_revision: str | None
    hf_config: str | None
    split: str
    size: int
    skip: int
    sampling: str
    shuffle_buffer_size: int | None
    row_adapter: str
    row_adapter_options: Mapping[str, Any]
    required_fields: Sequence[str]

    def as_key_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "hf_id": self.hf_id,
            "hf_revision": self.hf_revision,
            "hf_config": self.hf_config,
            "split": self.split,
            "size": int(self.size),
            "skip": int(self.skip),
            "sampling": self.sampling,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "row_adapter": self.row_adapter,
            "row_adapter_options": {str(k): v for k, v in sorted(self.row_adapter_options.items())},
            "required_fields": list(self.required_fields),
        }

    def key(self) -> str:
        canonical = json.dumps(
            self.as_key_payload(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class SnapshotManifest:
    """Sidecar describing a snapshot file.  Mismatch => cache miss, never a silent reuse."""

    key: str
    task_id: str
    license: str
    rows_written: int
    exhausted: bool
    created_utc: str
    datasets_version: str
    content_sha256: str
    spec: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def path_for(cache_dir: str | Path, task_id: str, spec: SnapshotSpec) -> Path:
    """``<cache_dir>/<task_id>/<upstream_split>-<key>.jsonl`` (FR-5.1)."""
    safe_split = str(spec.split).replace("/", "_")
    return Path(cache_dir) / task_id / f"{safe_split}-{spec.key()}.jsonl"


def manifest_path_for(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def restrict(row: Mapping[str, Any], required_fields: Sequence[str]) -> dict[str, Any]:
    """Keep only the fields the row adapter declared it reads (FR-5.3).

    Restricting at write time keeps the cache small *and* makes the key honest: a snapshot
    written for one adapter cannot accidentally satisfy another that reads more fields.
    """
    return {name: row[name] for name in required_fields if name in row}


def read(path: Path, spec: SnapshotSpec) -> tuple[list[dict[str, Any]], SnapshotManifest] | None:
    """Return cached rows + manifest, or ``None`` on absence or any manifest mismatch."""
    manifest_file = manifest_path_for(path)
    if not path.exists() or not manifest_file.exists():
        return None
    try:
        manifest_data = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None
    if manifest_data.get("key") != spec.key():
        return None
    # Compare in JSON form: the stored spec has been through a JSON round-trip, which turns
    # nested non-string mapping keys (IMDb's ``labels: {0: negative}``) into strings.  Comparing
    # against the raw payload made such a snapshot a permanent cache miss.
    expected = json.loads(json.dumps(spec.as_key_payload(), sort_keys=True, default=str))
    if manifest_data.get("spec") != expected:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if _content_sha256(rows) != manifest_data.get("content_sha256"):
        raise DataError(
            f"snapshot {path} does not match its manifest checksum; refusing to use a corrupted "
            f"or partially-written cache entry. Delete it and re-run online."
        )
    # A manifest may carry extra recorded measurements (over-length fraction, counts); keep only
    # the declared fields when rebuilding the dataclass so adding a measurement is never breaking.
    declared = {f.name for f in fields(SnapshotManifest)} - {"spec"}
    core = {key: value for key, value in manifest_data.items() if key in declared}
    return rows, SnapshotManifest(**core, spec=spec.as_key_payload())


def write(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: SnapshotSpec,
    task_id: str,
    license: str,
    exhausted: bool,
    extra: Mapping[str, Any] | None = None,
) -> SnapshotManifest:
    """Atomically write the JSONL rows and their manifest (tmp file + rename)."""
    try:
        import datasets

        datasets_version = str(datasets.__version__)
    except Exception:  # pragma: no cover - datasets is a hard dependency in practice
        datasets_version = "unknown"

    manifest = SnapshotManifest(
        key=spec.key(),
        task_id=task_id,
        license=license,
        rows_written=len(rows),
        exhausted=bool(exhausted),
        created_utc=datetime.now(timezone.utc).isoformat(),
        datasets_version=datasets_version,
        content_sha256=_content_sha256(rows),
        spec=spec.as_key_payload(),
    )
    payload = manifest.as_dict()
    if extra:
        payload.update({str(k): v for k, v in extra.items()})

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_rows = path.with_name(path.name + ".tmp")
    tmp_manifest = manifest_path_for(path).with_name(manifest_path_for(path).name + ".tmp")
    with tmp_rows.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n")
    with tmp_manifest.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp_rows, path)
    os.replace(tmp_manifest, manifest_path_for(path))
    return manifest
