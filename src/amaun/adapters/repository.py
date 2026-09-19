"""M3/M9 -- the adapter repository ``R_c`` and the candidate store.

Two design points carry scientific weight:

**1. Registration requires a capability token.**  ``register(..., authorised_by=decision)``
means a temporary adapter physically cannot enter ``R_c`` before the decision step
completes (master-prompt constraint 10 / idea.md S3).  The check is not a convention that a
future edit might overlook: without a ``Decision``-shaped object, the call raises.

**2. Archive is soft.**  Adapters are moved to ``archive/``, never deleted, so the growth
history stays auditable.  Brief S6 makes pruning optional; removing evidence of growth would
make the growth-control claim unfalsifiable.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.errors import (
    AdapterError,
    IncompatibleAdapterError,
    UnauthorisedRegistrationError,
)
from amaun.utils.logging import get_logger

log = get_logger("adapters.repository")


@runtime_checkable
class RegistrationAuthority(Protocol):
    """Structural type of an authorising decision.

    Kept as a Protocol so ``adapters`` does not import ``manager`` (that would create a
    dependency cycle and let the repository reach into decision logic).  Any object with a
    decision outcome and an id satisfies it.
    """

    outcome: Any
    decision_id: str


@dataclass
class AdapterEntry:
    """One repository slot: the delta, its metadata, and where it lives on disk."""

    delta: AdapterDelta
    metadata: AdapterMetadata
    path: Path | None = None

    @property
    def adapter_id(self) -> str:
        return self.metadata.adapter_id

    def memory_bytes(self) -> int:
        """``Mem_i = |phi_i| * b`` (idea.md S4.6), measured from real tensors."""
        return self.delta.storage_bytes()


class CandidateStore:
    """Scratch store for ``A_temp``.  Never part of ``R_c`` (idea.md S3)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, AdapterEntry] = {}

    def put(self, entry: AdapterEntry) -> AdapterEntry:
        path = self.root / entry.adapter_id
        path.mkdir(parents=True, exist_ok=True)
        entry.delta.save(path / "delta.pt")
        entry.metadata.content_hash = entry.delta.content_hash()
        entry.metadata.save(path / "metadata.json")
        entry.path = path
        self._entries[entry.adapter_id] = entry
        return entry

    def get(self, adapter_id: str) -> AdapterEntry:
        if adapter_id not in self._entries:
            raise AdapterError(f"no candidate '{adapter_id}' in the candidate store")
        return self._entries[adapter_id]

    def discard(self, adapter_id: str) -> None:
        """Drop a candidate (the REUSE outcome: ``A_temp`` is thrown away)."""
        entry = self._entries.pop(adapter_id, None)
        if entry is not None and entry.path is not None and entry.path.exists():
            shutil.rmtree(entry.path)

    def __contains__(self, adapter_id: object) -> bool:
        return adapter_id in self._entries

    def ids(self) -> list[str]:
        return sorted(self._entries)


class AdapterRepository:
    """``R_c`` -- the local adapter repository for one client.

    Supports the full brief-S6 surface: REGISTER, RETRIEVE, REUSE, FUSE, CREATE, EVALUATE,
    ARCHIVE, and (optional) COMPRESS.  ``FUSE`` and ``COMPRESS`` are driven from
    ``manager/``; the repository provides the state transitions and the audit journal.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        client_id: str = "c0",
        backbone_fingerprint: str | None = None,
        journal_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(exist_ok=True)
        self.client_id = client_id
        self.backbone_fingerprint = backbone_fingerprint
        self.journal_path = Path(journal_path) if journal_path else self.root / "repository_events.jsonl"
        self._entries: dict[str, AdapterEntry] = {}
        self._archived: dict[str, AdapterEntry] = {}
        self.candidates = CandidateStore(self.root / "_candidates")
        self._event_count = 0

    # -- basic surface -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, adapter_id: object) -> bool:
        return adapter_id in self._entries

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> list[AdapterEntry]:
        return [self._entries[i] for i in self.ids()]

    def __iter__(self) -> Iterator[AdapterEntry]:
        return iter(self.entries())

    @property
    def is_empty(self) -> bool:
        """idea.md S5.4 lines 2-4: an empty repository short-circuits to CREATE."""
        return not self._entries

    def memory_bytes(self) -> int:
        """``Mem(R_c) = sum_i Mem(A_i)`` (idea.md S7.1)."""
        return sum(entry.memory_bytes() for entry in self._entries.values())

    def total_parameters(self) -> int:
        return sum(entry.delta.parameter_count for entry in self._entries.values())

    def tasks_served(self) -> dict[str, list[str]]:
        """D-014 accounting: which tasks each adapter serves."""
        return {i: list(self._entries[i].metadata.tasks_served) for i in self.ids()}

    # -- journal -----------------------------------------------------------------
    def _journal(self, event: str, **payload: Any) -> dict[str, Any]:
        self._event_count += 1
        record = {
            "seq": self._event_count,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "client_id": self.client_id,
            "event": event,
            "repository_size": len(self._entries),
            "repository_bytes": self.memory_bytes(),
            **payload,
        }
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    def events(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        return [json.loads(l) for l in self.journal_path.read_text().splitlines() if l.strip()]

    # -- compatibility -----------------------------------------------------------
    def _check_compatible(self, metadata: AdapterMetadata) -> None:
        """Invariant I3.3: an adapter is only valid for the backbone it was fitted to."""
        if self.backbone_fingerprint is None:
            return
        if metadata.backbone_fingerprint != self.backbone_fingerprint:
            raise IncompatibleAdapterError(
                f"adapter '{metadata.adapter_id}' was fitted to backbone "
                f"{metadata.backbone_fingerprint[:12]} but this repository is bound to "
                f"{self.backbone_fingerprint[:12]}. Cross-run-mode checkpoint loading is "
                f"forbidden (idea.md S16.12)."
            )

    # -- REGISTER / CREATE -------------------------------------------------------
    def register(
        self,
        entry: AdapterEntry,
        *,
        authorised_by: RegistrationAuthority | None = None,
        bootstrap: bool = False,
        task_index: int | None = None,
    ) -> AdapterEntry:
        """Insert an adapter into ``R_c``.

        ``authorised_by`` is required unless ``bootstrap=True``.  ``bootstrap`` exists only
        for the ``repository_init: bootstrap_six`` protocol (D-002 O1), where the six seed
        adapters enter the repository unconditionally *by experimental design*, and it is
        journaled as such so a reviewer can see exactly which adapters bypassed the decision
        module and why.
        """
        if entry.adapter_id in self._entries:
            raise AdapterError(f"adapter '{entry.adapter_id}' is already registered")
        if not bootstrap and authorised_by is None:
            raise UnauthorisedRegistrationError(
                f"refusing to register '{entry.adapter_id}' without an authorising Decision. "
                f"A temporary adapter must stay out of R_c until the decision completes "
                f"(master_prompt constraint 10). Pass bootstrap=True only for the documented "
                f"seed-adapter protocol."
            )
        self._check_compatible(entry.metadata)

        path = self.root / entry.adapter_id
        path.mkdir(parents=True, exist_ok=True)
        entry.delta.save(path / "delta.pt")
        entry.metadata.content_hash = entry.delta.content_hash()
        entry.metadata.parameter_count = entry.delta.parameter_count
        entry.metadata.storage_bytes = entry.delta.storage_bytes()
        if task_index is not None and entry.metadata.created_at_task_index is None:
            entry.metadata.created_at_task_index = task_index
        entry.metadata.save(path / "metadata.json")
        entry.path = path
        self._entries[entry.adapter_id] = entry

        self.candidates.discard(entry.adapter_id)
        self._journal(
            "register",
            adapter_id=entry.adapter_id,
            origin=entry.metadata.origin,
            bootstrap=bootstrap,
            authorised_by=getattr(authorised_by, "decision_id", None),
            tasks_served=list(entry.metadata.tasks_served),
            parameter_count=entry.metadata.parameter_count,
        )
        return entry

    # -- RETRIEVE ----------------------------------------------------------------
    def retrieve(self, adapter_id: str) -> AdapterEntry:
        if adapter_id not in self._entries:
            raise AdapterError(f"adapter '{adapter_id}' is not in the repository")
        return self._entries[adapter_id]

    # -- REUSE -------------------------------------------------------------------
    def mark_reuse(self, adapter_id: str, task_id: str, task_index: int) -> AdapterEntry:
        """The REUSE outcome: an existing adapter serves a new task; ``R_c`` is unchanged."""
        entry = self.retrieve(adapter_id)
        entry.metadata.record_use(task_index)
        if task_id not in entry.metadata.tasks_served:
            entry.metadata.tasks_served.append(task_id)
        self._persist_metadata(entry)
        self._journal("reuse", adapter_id=adapter_id, task_id=task_id, task_index=task_index,
                      tasks_served=list(entry.metadata.tasks_served))
        return entry

    # -- FUSE --------------------------------------------------------------------
    def replace_with_fused(
        self,
        *,
        replaced_id: str,
        fused_entry: AdapterEntry,
        authorised_by: RegistrationAuthority,
        task_index: int | None = None,
    ) -> AdapterEntry:
        """``R'_c = (R_c \\ {A_i*}) union {A_merge}`` (idea.md S7.1).

        The replaced adapter is archived rather than deleted, so its contribution history
        remains auditable and the growth trajectory can be reconstructed.
        """
        replaced = self.retrieve(replaced_id)
        inherited = list(replaced.metadata.tasks_served)
        for task in inherited:
            if task not in fused_entry.metadata.tasks_served:
                fused_entry.metadata.tasks_served.append(task)
        fused_entry.metadata.merge_count = replaced.metadata.merge_count + 1
        fused_entry.metadata.performance_history = list(replaced.metadata.performance_history)
        if replaced.metadata.created_at_task_index is not None:
            fused_entry.metadata.created_at_task_index = replaced.metadata.created_at_task_index

        self.archive(replaced_id, reason=f"replaced by fused adapter {fused_entry.adapter_id}")
        registered = self.register(fused_entry, authorised_by=authorised_by, task_index=task_index)
        self._journal(
            "fuse_replace",
            replaced_id=replaced_id,
            fused_id=fused_entry.adapter_id,
            parents=list(fused_entry.metadata.parent_adapter_ids),
            tasks_served=list(registered.metadata.tasks_served),
            merge_count=registered.metadata.merge_count,
        )
        return registered

    def replace_pair_with_merged(
        self,
        *,
        first_id: str,
        second_id: str,
        merged_entry: AdapterEntry,
        authorised_by: RegistrationAuthority,
    ) -> AdapterEntry:
        """``R_c <- (R_c \\ {A_i*, A_j*}) union {A_merge_ij}`` -- compression (idea.md S7.2)."""
        for source_id in (first_id, second_id):
            source = self.retrieve(source_id)
            for task in source.metadata.tasks_served:
                if task not in merged_entry.metadata.tasks_served:
                    merged_entry.metadata.tasks_served.append(task)
        merged_entry.metadata.merge_count = (
            self.retrieve(first_id).metadata.merge_count
            + self.retrieve(second_id).metadata.merge_count
            + 1
        )
        self.archive(first_id, reason=f"compressed into {merged_entry.adapter_id}")
        self.archive(second_id, reason=f"compressed into {merged_entry.adapter_id}")
        registered = self.register(merged_entry, authorised_by=authorised_by)
        self._journal(
            "compress_merge",
            merged_pair=[first_id, second_id],
            merged_id=merged_entry.adapter_id,
            tasks_served=list(registered.metadata.tasks_served),
        )
        return registered

    # -- EVALUATE ----------------------------------------------------------------
    def record_evaluation(
        self, adapter_id: str, task_id: str, metrics: dict[str, Any], task_index: int
    ) -> AdapterEntry:
        entry = self.retrieve(adapter_id)
        entry.metadata.record_performance(task_id, metrics, task_index)
        entry.metadata.performance_metrics = dict(metrics)
        self._persist_metadata(entry)
        self._journal("evaluate", adapter_id=adapter_id, task_id=task_id, metrics=metrics)
        return entry

    # -- ARCHIVE -----------------------------------------------------------------
    def archive(self, adapter_id: str, *, reason: str) -> AdapterEntry:
        """Soft removal: move to ``archive/``.  Never deletes (growth stays auditable)."""
        entry = self._entries.pop(adapter_id)
        entry.metadata.archived = True
        entry.metadata.archived_reason = reason
        destination = self.root / "archive" / adapter_id
        if entry.path is not None and entry.path.exists():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(entry.path), str(destination))
        entry.path = destination
        self._persist_metadata(entry)
        self._archived[adapter_id] = entry
        self._journal("archive", adapter_id=adapter_id, reason=reason)
        return entry

    def archived_ids(self) -> list[str]:
        return sorted(self._archived)

    # -- snapshots ---------------------------------------------------------------
    def snapshot(self, path: Path | str, *, label: str = "") -> Path:
        """Persist a full repository state summary for later analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": label,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "client_id": self.client_id,
            "backbone_fingerprint": self.backbone_fingerprint,
            "size": len(self._entries),
            "memory_bytes": self.memory_bytes(),
            "total_parameters": self.total_parameters(),
            "adapters": [
                {
                    **entry.metadata.as_dict(),
                    "delta_summary": entry.delta.summary(),
                }
                for entry in self.entries()
            ],
            "archived": sorted(self._archived),
        }
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        return path

    def _persist_metadata(self, entry: AdapterEntry) -> None:
        if entry.path is not None:
            entry.metadata.save(entry.path / "metadata.json")

    # -- loading -----------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | str, **kwargs: Any) -> "AdapterRepository":
        """Rebuild a repository from disk (checkpoint restoration)."""
        repo = cls(root, **kwargs)
        for child in sorted(Path(root).iterdir()):
            if not child.is_dir() or child.name in {"archive", "_candidates"}:
                continue
            metadata_path = child / "metadata.json"
            delta_path = child / "delta.pt"
            if not (metadata_path.exists() and delta_path.exists()):
                continue
            metadata = AdapterMetadata.load(metadata_path)
            delta = AdapterDelta.load(delta_path)
            if metadata.content_hash and delta.content_hash() != metadata.content_hash:
                raise AdapterError(
                    f"content hash mismatch for '{metadata.adapter_id}': stored weights do not "
                    f"match the recorded hash (corrupt or tampered checkpoint)"
                )
            repo._entries[metadata.adapter_id] = AdapterEntry(delta=delta, metadata=metadata, path=child)
        log.info("loaded repository from %s with %d adapters", root, len(repo))
        return repo
