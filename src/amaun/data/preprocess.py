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


PROMPT_TRUNCATION_POLICIES = ("head", "middle")

# Sentinel used to split a chat template into the text before and after the user content.
_CONTENT_SENTINEL = "\u0000AMAUN_CONTENT\u0000"


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int,
    use_chat_template: bool = True,
    truncation: str = "head",
) -> list[int]:
    """Token ids of the (templated) prompt, at most ``max_input_tokens`` long.

    ``head`` is the original behaviour: encode the templated prompt and keep its first
    ``max_input_tokens`` tokens.  For a long input that drops the END of the prompt -- the
    question in a QA template and the chat template's assistant header -- so the model is asked
    to continue a truncated context with no question and no turn boundary.

    ``middle`` keeps the template's prefix and suffix intact and cuts the *user content* in the
    middle: the first three quarters of the content budget come from its head (instruction and
    the start of the context) and the last quarter from its tail (the question and answer cue).
    The policy is a per-task config value (``datasets.tasks.<id>.prompt_truncation``) because it
    changes what the model sees.
    """
    if truncation not in PROMPT_TRUNCATION_POLICIES:
        raise ValueError(
            f"unknown prompt truncation '{truncation}'; expected one of {PROMPT_TRUNCATION_POLICIES}"
        )
    if truncation == "head":
        prompt_text = build_chat_prompt(tokenizer, prompt) if use_chat_template else prompt
        return _encode(tokenizer, prompt_text)[:max_input_tokens]

    if use_chat_template:
        templated = build_chat_prompt(tokenizer, _CONTENT_SENTINEL)
        prefix, _, suffix = templated.partition(_CONTENT_SENTINEL)
    else:
        prefix, suffix = "", ""
    prefix_ids = _encode(tokenizer, prefix) if prefix else []
    suffix_ids = _encode(tokenizer, suffix) if suffix else []
    content_ids = _encode(tokenizer, prompt)
    budget = max_input_tokens - len(prefix_ids) - len(suffix_ids)
    if budget <= 0:
        # The template alone exceeds the limit; fall back to a hard cut so the bound still holds.
        return (prefix_ids + content_ids + suffix_ids)[:max_input_tokens]
    if len(content_ids) > budget:
        tail = budget // 4
        head = budget - tail
        content_ids = content_ids[:head] + (content_ids[-tail:] if tail else [])
    return prefix_ids + content_ids + suffix_ids


def tokenize_example(
    tokenizer: Any,
    example: Example,
    *,
    max_input_tokens: int,
    max_target_tokens: int,
    use_chat_template: bool = True,
    prompt_truncation: str = "head",
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` with prompt positions masked out."""
    prompt_ids = encode_prompt(
        tokenizer, example.prompt, max_input_tokens=max_input_tokens,
        use_chat_template=use_chat_template, truncation=prompt_truncation,
    )
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
    prompt_truncation: str = "head",
) -> TokenizedBatch:
    """Right-pad a batch.  Padding positions are masked in both attention and labels."""
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    encoded = [
        tokenize_example(
            tokenizer, example,
            max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens,
            use_chat_template=use_chat_template,
            prompt_truncation=prompt_truncation,
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

    Only response positions are gathered before the fp32 upcast.  Upcasting the whole
    ``[batch, seq, vocab]`` tensor first (the previous form) allocated a full fp32 copy of the
    logits -- ~1.6 GB at batch 4 x 640 tokens x 151 936 vocab -- although prompt positions
    contribute nothing to the loss.  The value is identical: the same positions, the same fp32
    cross-entropy, the same mean.
    """
    shift_labels = labels[:, 1:]
    keep = shift_labels != IGNORE_INDEX
    selected = logits[:, :-1, :][keep]   # empty when nothing is kept -> NaN, as before
    return torch.nn.functional.cross_entropy(selected.float(), shift_labels[keep])


def response_mask(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask of response positions, shifted to align with prediction positions."""
    return (labels[:, 1:] != IGNORE_INDEX)
