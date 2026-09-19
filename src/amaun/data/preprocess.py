"""M2 -- tokenisation and response masking.

The masking rule is load-bearing.  Five of the six tasks are open-ended generation, so
computing loss over prompt tokens would train the adapter to model the *inputs* as well as
the responses -- a different objective from the one the research claims to study
(assumption A-03).  Prompt positions get label ``-100`` and contribute exactly zero loss,
and ``tests/test_loss_masking.py`` asserts it numerically rather than by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from amaun.data.splits import Example

IGNORE_INDEX = -100


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_lengths: list[int]

    def to(self, device: str | torch.device) -> "TokenizedBatch":
        return TokenizedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            prompt_lengths=self.prompt_lengths,
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Format a prompt with the backbone's chat template where one exists.

    Instruct-tuned backbones expect their own template; ignoring it would systematically
    handicap every arm, including the baselines.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - templates vary
            pass
    return prompt


def tokenize_example(
    tokenizer: Any,
    example: Example,
    *,
    max_input_tokens: int,
    max_target_tokens: int,
    use_chat_template: bool = True,
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` with prompt positions masked out."""
    prompt_text = build_chat_prompt(tokenizer, example.prompt) if use_chat_template else example.prompt
    prompt_ids = _encode(tokenizer, prompt_text)[:max_input_tokens]
    target_ids = _encode(tokenizer, example.target)[:max_target_tokens]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        target_ids = target_ids + [eos]
    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
    return input_ids, labels


def collate(
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    max_input_tokens: int,
    max_target_tokens: int,
    use_chat_template: bool = True,
) -> TokenizedBatch:
    """Right-pad a batch.  Padding positions are masked in both attention and labels."""
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    encoded = [
        tokenize_example(
            tokenizer, example,
            max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens,
            use_chat_template=use_chat_template,
        )
        for example in examples
    ]
    width = max(len(ids) for ids, _ in encoded)
    input_ids, attention, labels, prompt_lengths = [], [], [], []
    for ids, label in encoded:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(label + [IGNORE_INDEX] * pad)
        prompt_lengths.append(sum(1 for l in label if l == IGNORE_INDEX))
    return TokenizedBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        prompt_lengths=prompt_lengths,
    )


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token CE over non-ignored positions only.

    ``logits[:, :-1]`` predicts ``labels[:, 1:]`` -- the standard causal shift, written
    explicitly here rather than relying on a model's internal loss, because the KD path
    needs the same shift applied to teacher and student logits and must be provably
    identical to the task loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def response_mask(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask of response positions, shifted to align with prediction positions."""
    return (labels[:, 1:] != IGNORE_INDEX)
