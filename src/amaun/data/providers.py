"""M2 -- dataset providers.

Two providers:

- ``huggingface``: loads a *pinned* dataset (id + revision).  Refuses unpinned rows
  (D-006).  Never downloads more than the configured volume: we stream and stop, because
  brief S4 explicitly forbids blindly processing entire datasets.
- ``synthetic``: deterministic offline pseudo-data with the right *shape* per task type.
  This is what lets ``pytest`` and the ``dummy`` tier exercise the full pipeline with no
  network, which idea.md S14 requires before any expensive run.

The synthetic provider produces **structurally** realistic data (a context and a
question-shaped target, of plausible lengths) but its content is meaningless.  It is a
correctness fixture, never a source of a reported number.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator, Sequence

from amaun.data.registry import TaskSpec, TaskType
from amaun.data.splits import Example
from amaun.errors import DataError
from amaun.utils.logging import get_logger

log = get_logger("data.providers")

_VOCAB = (
    "market revenue quarter policy patient dataset gradient adapter summary dialogue "
    "question answer finance report growth decline analysis model token sequence layer "
    "client server round federated continual forgetting utility fusion threshold"
).split()


def _rng_for(task_id: str, index: int, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{task_id}|{index}|{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(count))


def _synthetic_example(spec: TaskSpec, index: int, seed: int) -> Example:
    """One deterministic pseudo-example whose shape matches the task type."""
    rng = _rng_for(spec.task_id, index, seed)
    if spec.task_type is TaskType.SUMMARIZATION:
        fields = {"article": _words(rng, rng.randint(60, 120))}
        target = _words(rng, rng.randint(10, 20))
    elif spec.task_type is TaskType.QUESTION_ANSWERING:
        context = _words(rng, rng.randint(40, 80))
        answer = rng.choice(context.split())
        fields = {"context": context, "question": f"what is {answer} about"}
        target = answer
    elif spec.task_type is TaskType.INSTRUCTION_FOLLOWING:
        fields = {"instruction": f"explain {_words(rng, 3)}", "input": _words(rng, rng.randint(5, 20))}
        target = _words(rng, rng.randint(15, 30))
    elif spec.task_type is TaskType.DIALOGUE:
        fields = {"history": " | ".join(_words(rng, rng.randint(4, 10)) for _ in range(rng.randint(2, 4)))}
        target = _words(rng, rng.randint(6, 15))
    elif spec.task_type is TaskType.LONG_FORM_GENERATION:
        fields = {"question": f"why does {_words(rng, 3)} happen"}
        target = _words(rng, rng.randint(40, 90))
    else:  # DOMAIN_GENERATION
        fields = {"context": _words(rng, rng.randint(30, 60)), "question": f"compute {_words(rng, 2)}"}
        target = _words(rng, rng.randint(8, 25))

    prompt = render_prompt(spec, fields)
    return Example(prompt=prompt, target=target, task_id=spec.task_id,
                   meta={"provider": "synthetic", "index": index, **fields})


def render_prompt(spec: TaskSpec, fields: dict[str, Any]) -> str:
    """Fill the task's prompt template.

    Missing fields fail loudly rather than rendering an empty slot, because a silently
    truncated prompt changes the task the adapter learns.
    """
    try:
        return spec.prompt_template.format(**fields)
    except KeyError as exc:
        raise DataError(
            f"task '{spec.task_id}' prompt_template needs field {exc} but the example "
            f"provides {sorted(fields)}"
        ) from exc


def synthetic_pool(spec: TaskSpec, size: int, *, seed: int, offset: int = 0) -> list[Example]:
    return [_synthetic_example(spec, offset + i, seed) for i in range(size)]


def huggingface_pool(
    spec: TaskSpec,
    size: int,
    *,
    split: str,
    cache_dir: str | None = None,
    streaming: bool = True,
) -> list[Example]:
    """Load at most ``size`` examples from a pinned HF dataset split.

    Streaming by default so we never materialise a full corpus (brief S4: "Do NOT blindly
    download or process entire datasets").
    """
    spec.require_pinned()
    from datasets import load_dataset

    log.info(
        "loading %s (revision=%s, config=%s) split=%s, cap=%d",
        spec.hf_id, spec.hf_revision, spec.hf_config, split, size,
    )
    dataset = load_dataset(
        spec.hf_id,
        spec.hf_config,
        split=split,
        revision=spec.hf_revision,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    examples: list[Example] = []
    for row in dataset:
        if len(examples) >= size:
            break
        fields = {name: row[name] for name in spec.input_fields if name in row}
        if len(fields) != len(spec.input_fields):
            missing = set(spec.input_fields) - set(fields)
            raise DataError(f"dataset {spec.hf_id} row lacks input fields {sorted(missing)}")
        target = row.get(spec.target_field)
        if target is None:
            raise DataError(f"dataset {spec.hf_id} row lacks target field '{spec.target_field}'")
        if isinstance(target, dict) and "text" in target:  # e.g. SQuAD answers
            texts = target["text"]
            target = texts[0] if texts else ""
        if isinstance(target, list):
            target = target[0] if target else ""
        examples.append(
            Example(
                prompt=render_prompt(spec, fields),
                target=str(target).strip(),
                task_id=spec.task_id,
                meta={"provider": "huggingface", "hf_id": spec.hf_id, "revision": spec.hf_revision},
            )
        )
    if not examples:
        raise DataError(f"loaded zero examples for '{spec.task_id}' from {spec.hf_id}:{split}")
    return examples
