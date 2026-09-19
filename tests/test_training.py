"""M4 -- training loop, loss masking, adapter independence (criteria C3, C4, C8)."""

import pytest
import torch

from amaun.adapters.lora import ensure_lora, trainable_parameters
from amaun.data.preprocess import IGNORE_INDEX, collate, masked_cross_entropy, tokenize_example
from amaun.data.splits import trainer_guard
from amaun.models.backbone import backbone_fingerprint
from amaun.training.budget import ArmBudget
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.seed import derive_seed


def _train(cfg, backbone, task, splits, lora_spec, adapter_id, budget=None):
    return train_adapter(
        backbone=backbone, task=task, guard=trainer_guard(splits[task.task_id]),
        lora_spec=lora_spec, train_cfg=TrainingConfig.from_config(cfg.section("training")),
        resolved=cfg, adapter_id=adapter_id, budget=budget,
    )


def test_optimizer_sees_only_adapter_parameters(session_cfg, backbone, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    params = trainable_parameters(model)
    assert params
    adapter_ids = {
        id(p) for n, p in model.named_parameters()
        if "lora_" in n.lower() or "fusion_logit" in n.lower()
    }
    assert {id(p) for p in params} <= adapter_ids


def test_training_leaves_the_backbone_bit_identical(session_cfg, backbone, tasks, splits, lora_spec):
    before = backbone_fingerprint(backbone.model)
    _train(session_cfg, backbone, tasks[0], splits, lora_spec, "frozen-check")
    assert backbone_fingerprint(backbone.model) == before


def test_training_actually_updates_the_adapter(session_cfg, backbone, tasks, splits, lora_spec):
    delta, record = _train(session_cfg, backbone, tasks[0], splits, lora_spec, "moves")
    assert record.steps > 0
    # LoRA starts with B = 0, so a trained adapter must have a non-zero effective update.
    assert any(float(layer.B.abs().sum()) > 0 for _key, layer in delta.items())


def test_budget_is_recorded(session_cfg, backbone, tasks, splits, lora_spec):
    budget = ArmBudget(arm_id="test")
    _train(session_cfg, backbone, tasks[0], splits, lora_spec, "budget", budget=budget)
    assert budget.optimizer_steps > 0
    assert budget.tokens_seen > 0
    assert budget.examples_seen > 0


def test_prompt_tokens_contribute_zero_loss(backbone, tasks, splits):
    """Assumption A-03 / criterion C8, verified numerically rather than by inspection."""
    from amaun.data.splits import SplitName

    task = tasks[0]
    examples = splits[task.task_id][SplitName.TRAIN][:2]
    batch = collate(
        backbone.tokenizer, examples,
        max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
    )
    assert (batch.labels == IGNORE_INDEX).any(), "no masked positions at all"

    with torch.no_grad():
        logits = backbone.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits
    reference = masked_cross_entropy(logits, batch.labels)

    mask = (batch.labels == IGNORE_INDEX)

    # Alignment matters: masked_cross_entropy pairs logits[:, :-1] with labels[:, 1:], so the
    # logit position that predicts label `p` is `p-1`. A test that perturbs logits at masked
    # LABEL positions would be off by one and could touch a response prediction.
    ignored_logit_positions = mask[:, 1:]          # [batch, seq-1]
    response_logit_positions = ~ignored_logit_positions

    def perturb_single_logit(where: torch.Tensor) -> torch.Tensor:
        """Add 100 to vocabulary index 0 at the selected logit positions.

        A single entry, not the whole vocabulary: adding a constant across all logits leaves
        softmax unchanged and would make this test vacuous.
        """
        out = logits.clone()
        padded = torch.zeros(logits.shape[:2], dtype=torch.bool)
        padded[:, :-1] = where
        out[..., 0] = out[..., 0] + 100.0 * padded.to(out.dtype)
        return out

    assert torch.allclose(
        masked_cross_entropy(perturb_single_logit(ignored_logit_positions), batch.labels),
        reference, atol=1e-5,
    ), "prompt/padding positions influenced the loss"

    assert response_logit_positions.any(), "batch contains no response positions to check"
    assert not torch.allclose(
        masked_cross_entropy(perturb_single_logit(response_logit_positions), batch.labels),
        reference, atol=1e-5,
    ), "response positions did NOT influence the loss -- the mask is inverted"


def _first_vocab_mask(logits: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting vocabulary index 0 at every position."""
    out = torch.zeros_like(logits, dtype=torch.bool)
    out[..., 0] = True
    return out


def test_tokenize_masks_exactly_the_prompt(backbone, tasks, splits):
    from amaun.data.splits import SplitName

    task = tasks[0]
    example = splits[task.task_id][SplitName.TRAIN][0]
    input_ids, labels = tokenize_example(
        backbone.tokenizer, example,
        max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
    )
    assert len(input_ids) == len(labels)
    n_masked = sum(1 for label in labels if label == IGNORE_INDEX)
    assert 0 < n_masked < len(labels)
    # Unmasked labels must equal the corresponding inputs (teacher forcing on the response).
    for token, label in zip(input_ids, labels):
        if label != IGNORE_INDEX:
            assert token == label


def test_derived_seeds_are_stable_and_distinct():
    assert derive_seed(1234, "A1") == derive_seed(1234, "A1")
    assert derive_seed(1234, "A1") != derive_seed(1234, "A2")
    assert derive_seed(1234, "A1") != derive_seed(1235, "A1")
