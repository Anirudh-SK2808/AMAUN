"""M2 -- the task/dataset registry.

Implements idea.md S0.2 (the LoRA-1..6 curriculum) as a *config-driven, extensible* table,
with every dataset's identity, revision, and license recorded.

Two rows are deliberately left unpinned (``hf_id: None``): the "FLAN-style" instruction row
and the ELI5 long-form row.  See docs/decision_register.md D-006 -- "FLAN-style" is not a
dataset, and the original ELI5 HF dataset was withdrawn over Reddit licensing.  Guessing a
substitute would silently change what the adapter *is*, so the loader raises
:class:`UnpinnedDatasetError` instead.  The ``synthetic`` provider keeps all other work
unblocked in the meantime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any

from amaun.errors import DataError, UnpinnedDatasetError


class TaskType(str, enum.Enum):
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    INSTRUCTION_FOLLOWING = "instruction_following"
    DIALOGUE = "dialogue"
    LONG_FORM_GENERATION = "long_form_generation"
    DOMAIN_GENERATION = "domain_generation"


@dataclass(frozen=True)
class TaskSpec:
    """Everything that defines a task.  Built from ``configs/datasets/*.yaml``."""

    task_id: str
    task_name: str
    task_type: TaskType
    description: str
    hf_id: str | None
    hf_revision: str | None
    hf_config: str | None
    license: str
    license_notes: str
    primary_score: str          # D-009: the single scalar used by decisions and CL metrics
    metrics: tuple[str, ...]
    input_fields: tuple[str, ...]
    target_field: str
    prompt_template: str
    max_input_tokens: int
    max_target_tokens: int
    upstream_test_source: str   # documents which upstream split becomes our TEST (protocol S2)
    pinned: bool = True
    unpinned_reason: str | None = None

    def require_pinned(self) -> None:
        if not self.pinned or self.hf_id is None:
            raise UnpinnedDatasetError(
                f"task '{self.task_id}' has no pinned dataset. {self.unpinned_reason or ''} "
                f"Resolve docs/decision_register.md D-006 and set hf_id/hf_revision in "
                f"configs/datasets/. AMAUN will not guess a dataset, because the choice "
                f"changes what the adapter represents."
            )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        return data


def task_spec_from_config(task_id: str, section: dict[str, Any]) -> TaskSpec:
    required = (
        "task_name", "task_type", "primary_score", "metrics", "prompt_template",
        "target_field", "input_fields", "max_input_tokens", "max_target_tokens",
    )
    missing = [key for key in required if key not in section]
    if missing:
        raise DataError(f"dataset config for '{task_id}' is missing keys: {missing}")
    hf_id = section.get("hf_id")
    return TaskSpec(
        task_id=task_id,
        task_name=str(section["task_name"]),
        task_type=TaskType(section["task_type"]),
        description=str(section.get("description", "")),
        hf_id=hf_id,
        hf_revision=section.get("hf_revision"),
        hf_config=section.get("hf_config"),
        license=str(section.get("license", "UNKNOWN")),
        license_notes=str(section.get("license_notes", "")),
        primary_score=str(section["primary_score"]),
        metrics=tuple(section["metrics"]),
        input_fields=tuple(section["input_fields"]),
        target_field=str(section["target_field"]),
        prompt_template=str(section["prompt_template"]),
        max_input_tokens=int(section["max_input_tokens"]),
        max_target_tokens=int(section["max_target_tokens"]),
        upstream_test_source=str(section.get("upstream_test_source", "UNDOCUMENTED")),
        pinned=bool(hf_id) and bool(section.get("pinned", True)),
        unpinned_reason=section.get("unpinned_reason"),
    )


def load_curriculum(cfg_section: dict[str, Any]) -> list[TaskSpec]:
    """Build the ordered task curriculum from config.

    Order matters (it is the task arrival order) and is therefore explicit in the config
    rather than implied by dict iteration order.
    """
    tasks = cfg_section.get("tasks")
    order = cfg_section.get("order")
    if not tasks:
        raise DataError("dataset config has no 'tasks' mapping")
    if not order:
        raise DataError(
            "dataset config has no 'order' list; task arrival order is an experimental "
            "variable and must be stated explicitly (idea.md S0.2)"
        )
    unknown = [t for t in order if t not in tasks]
    if unknown:
        raise DataError(f"curriculum order references unknown tasks: {unknown}")
    return [task_spec_from_config(task_id, tasks[task_id]) for task_id in order]
