"""M2 -- batching over guarded splits.

Every loader is constructed from a :class:`SplitAccessGuard`, so obtaining batches requires
having been granted access to that split.  There is no constructor that takes raw examples
plus a split name, which is what keeps the guard from being incidental.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from amaun.data.preprocess import TokenizedBatch, collate
from amaun.data.registry import TaskSpec
from amaun.data.splits import Example, SplitAccessGuard, SplitName


@dataclass
class BatchPlan:
    """Batching configuration.  Values come from config, never from defaults here."""

    batch_size: int
    shuffle: bool
    seed: int
    drop_last: bool = False


class GuardedLoader:
    """Deterministic batch iterator over one permitted split."""

    def __init__(
        self,
        guard: SplitAccessGuard,
        split: SplitName | str,
        spec: TaskSpec,
        tokenizer: Any,
        plan: BatchPlan,
        *,
        use_chat_template: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.guard = guard
        self.split = SplitName(split)
        self.spec = spec
        self.tokenizer = tokenizer
        self.plan = plan
        self.use_chat_template = use_chat_template
        self._examples: list[Example] = list(guard.get(self.split))
        if max_examples is not None:
            self._examples = self._examples[:max_examples]

    def __len__(self) -> int:
        count = len(self._examples)
        if self.plan.drop_last:
            return count // self.plan.batch_size
        return (count + self.plan.batch_size - 1) // self.plan.batch_size

    @property
    def num_examples(self) -> int:
        return len(self._examples)

    def examples(self) -> list[Example]:
        return list(self._examples)

    def __iter__(self) -> Iterator[TokenizedBatch]:
        order = list(range(len(self._examples)))
        if self.plan.shuffle:
            random.Random(self.plan.seed).shuffle(order)
        for start in range(0, len(order), self.plan.batch_size):
            chunk = order[start : start + self.plan.batch_size]
            if self.plan.drop_last and len(chunk) < self.plan.batch_size:
                continue
            yield collate(
                self.tokenizer,
                [self._examples[i] for i in chunk],
                max_input_tokens=self.spec.max_input_tokens,
                max_target_tokens=self.spec.max_target_tokens,
                use_chat_template=self.use_chat_template,
            )


def fixed_reference_batch(
    guard: SplitAccessGuard,
    split: SplitName | str,
    spec: TaskSpec,
    tokenizer: Any,
    *,
    size: int,
    seed: int,
    use_chat_template: bool = True,
) -> TokenizedBatch:
    """The **fixed reference minibatch** of idea.md S4.3.

    Gradient-alignment features are only comparable if both adapters are differentiated
    against the *same* batch.  Selection is a deterministic function of ``(task, seed)``, so
    the same reference batch is reproduced across adapters, across candidate pairs, and
    across processes.
    """
    examples = list(guard.get(SplitName(split)))
    order = sorted(range(len(examples)), key=lambda i: examples[i].example_id)
    picked = [examples[i] for i in random.Random(seed).sample(order, k=min(size, len(order)))]
    return collate(
        tokenizer, picked,
        max_input_tokens=spec.max_input_tokens,
        max_target_tokens=spec.max_target_tokens,
        use_chat_template=use_chat_template,
    )
