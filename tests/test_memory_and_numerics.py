"""Memory-motivated rewrites must not change any value they compute.

Each test pins a rewrite against the formula it replaced: response-only cross-entropy, the
compact KD teacher cache, and token-weighted batched validation loss.  Also covers the prompt
truncation policy, exact split caps, the one-class threshold policy, and the IMDb row adapter.
"""

from __future__ import annotations

import pytest
import torch

from amaun.data.preprocess import (
    IGNORE_INDEX,
    TokenizedBatch,
    collate,
    encode_prompt,
    masked_cross_entropy,
    response_mask,
)
from amaun.data.registry import task_spec_from_config
from amaun.data.row_adapters import ROW_ADAPTERS
from amaun.data.splits import Example, SplitName, SplitRatios, build_splits
from amaun.errors import DataError
from amaun.manager.threshold import select_thresholds
from amaun.training.kd import capture_teacher, kd_loss


def _labels(batch=2, seq=9, n_prompt=4, vocab=50):
    labels = torch.full((batch, seq), IGNORE_INDEX)
    labels[:, n_prompt:] = torch.randint(1, vocab, (batch, seq - n_prompt))
    labels[1, -2:] = IGNORE_INDEX           # ragged: padding at the end of one row
    return labels


def _reference_ce(logits, labels):
    """The original implementation: full fp32 copy, then ignore_index."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def test_response_only_cross_entropy_matches_the_original_and_keeps_gradients():
    torch.manual_seed(0)
    logits = torch.randn(2, 9, 50, requires_grad=True)
    labels = _labels()
    new = masked_cross_entropy(logits, labels)
    old = _reference_ce(logits, labels)
    assert torch.allclose(new, old, atol=1e-6)
    (grad_new,) = torch.autograd.grad(new, logits)
    (grad_old,) = torch.autograd.grad(old, logits)
    assert torch.allclose(grad_new, grad_old, atol=1e-6)


def _reference_kd(student, teacher, labels, temperature):
    """The original kd_loss over full tensors, masked after the softmaxes."""
    s, t = student[:, :-1, :], teacher[:, :-1, :]
    mask = response_mask(labels)
    log_s = torch.log_softmax(s.float() / temperature, dim=-1)
    p_t = torch.softmax(t.float() / temperature, dim=-1)
    log_t = torch.log_softmax(t.float() / temperature, dim=-1)
    per = (p_t * (log_t - log_s)).sum(-1) * mask
    return temperature**2 * per.sum() / mask.sum()


def test_compact_and_full_teacher_give_the_reference_kd_loss():
    torch.manual_seed(1)
    student = torch.randn(2, 9, 50)
    teacher = torch.randn(2, 9, 50)
    labels = _labels()
    reference = _reference_kd(student, teacher, labels, 2.0)
    compact = teacher[:, :-1, :][response_mask(labels)]
    assert torch.allclose(kd_loss(student, teacher, labels, temperature=2.0), reference, atol=1e-6)
    assert torch.allclose(kd_loss(student, compact, labels, temperature=2.0), reference, atol=1e-6)


def test_capture_teacher_caches_only_response_positions():
    class _Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            torch.manual_seed(2)
            return type("Out", (), {"logits": torch.randn(*input_ids.shape, 50)})()

        def eval(self):
            return self

    labels = _labels()
    batch = TokenizedBatch(input_ids=torch.randint(1, 50, labels.shape),
                           attention_mask=torch.ones_like(labels), labels=labels, prompt_lengths=[4, 4])
    cache = capture_teacher(_Model(), [batch], teacher_adapter_id="A_old", temperature=2.0, device="cpu")
    assert cache.logits[0].shape == (int(response_mask(labels).sum()), 50)


def test_batched_validation_loss_equals_the_single_pass_token_mean(backbone, tasks, splits, lora_spec):
    from amaun.adapters.lora import ensure_lora, extract_delta
    from amaun.manager.bootstrap import _validation_loss

    model = ensure_lora(backbone.model, lora_spec)
    delta = extract_delta(model, adapter_id="probe")
    task = tasks[0]
    examples = list(splits[task.task_id][SplitName.VALIDATION])[:5]
    batch = collate(backbone.tokenizer, examples, max_input_tokens=task.max_input_tokens,
                    max_target_tokens=task.max_target_tokens)
    with torch.no_grad():
        single = float(masked_cross_entropy(
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits, batch.labels))
    batched = _validation_loss(model, delta, task, splits[task.task_id], backbone.tokenizer, "cpu",
                               max_examples=5, batch_size=2)
    assert batched == pytest.approx(single, rel=1e-5)


# --------------------------------------------------------------------------------------
# prompt truncation
# --------------------------------------------------------------------------------------


def test_head_truncation_is_the_original_behaviour(backbone):
    tokenizer = backbone.tokenizer
    prompt = "x" * 500 + " QUESTION?"
    templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True)
    assert encode_prompt(tokenizer, prompt, max_input_tokens=100) == tokenizer.encode(templated)[:100]


def test_middle_truncation_keeps_the_question_and_the_assistant_header(backbone):
    tokenizer = backbone.tokenizer
    prompt = "Context: " + "y" * 800 + "\nQuestion: what is it?\nAnswer:"
    ids = encode_prompt(tokenizer, prompt, max_input_tokens=120, truncation="middle")
    text = tokenizer.decode(ids)
    assert len(ids) <= 120
    assert text.endswith("<|assistant|>")
    assert "Answer:" in text and text.lstrip().startswith("<|user|>Context:")
    # A prompt that already fits is untouched by either policy.
    short = "short prompt"
    assert encode_prompt(tokenizer, short, max_input_tokens=120, truncation="middle") == \
        encode_prompt(tokenizer, short, max_input_tokens=120, truncation="head")


# --------------------------------------------------------------------------------------
# split caps, thresholds, IMDb rows
# --------------------------------------------------------------------------------------


def _pool(n, task_id="t"):
    return [Example(prompt=f"p{i}", target=f"t{i}", task_id=task_id) for i in range(n)]


def test_split_caps_are_exact_deterministic_and_strict():
    ratios = SplitRatios(train=0.7, validation=0.15, support=0.15)
    test_pool = [Example(prompt=f"q{i}", target="x", task_id="t") for i in range(50)]
    splits = build_splits("t", training_pool=_pool(400), test_pool=test_pool, seed=3, ratios=ratios)
    capped = splits.cap({"train": 200, "test": 30}, seed=3, strict=True)
    again = splits.cap({"train": 200, "test": 30}, seed=3, strict=True)
    assert len(capped[SplitName.TRAIN]) == 200 and len(capped[SplitName.TEST]) == 30
    assert capped.hashes(SplitName.TRAIN) == again.hashes(SplitName.TRAIN)
    capped.assert_disjoint()
    with pytest.raises(DataError, match="smaller than data.split_caps"):
        splits.cap({"train": 10_000}, seed=3, strict=True)
    assert len(splits.cap({"train": 10_000}, seed=3, strict=False)[SplitName.TRAIN]) < 10_000


def test_conservative_one_class_policy_refuses_merges_when_every_observation_was_harmful():
    utilities = [0.1, 0.4, 0.2]
    median = select_thresholds({"selection": "dev_stream"}, observed_utilities=utilities,
                               beneficial_flags=[False] * 3)
    assert median.tau_u == pytest.approx(0.2)            # original behaviour preserved by default
    conservative = select_thresholds({"selection": "dev_stream", "one_class_policy": "conservative"},
                                     observed_utilities=utilities, beneficial_flags=[False] * 3)
    assert conservative.tau_u > max(utilities)
    only_good = select_thresholds({"selection": "dev_stream", "one_class_policy": "conservative"},
                                  observed_utilities=utilities, beneficial_flags=[True] * 3)
    assert only_good.tau_u == pytest.approx(min(utilities))


def test_label_to_text_maps_imdb_labels_and_refuses_unknown_ones(raw_config):
    section = dict(raw_config["datasets"]["tasks"]["sentiment"])
    spec = task_spec_from_config("sentiment", section)
    adapter = ROW_ADAPTERS["label_to_text"]
    [(fields, target)] = adapter.adapt(spec, {"text": "great film", "label": 1}, spec.row_adapter_options)
    assert fields == {"text": "great film"} and target == "positive"
    assert adapter.adapt(spec, {"text": "dull", "label": 0}, spec.row_adapter_options)[0][1] == "negative"
    with pytest.raises(DataError):
        adapter.adapt(spec, {"text": "?", "label": 7}, spec.row_adapter_options)


@pytest.mark.parametrize("d_out,d_in,k,target", [(896, 1536, 32, 16), (300, 200, 8, 16), (64, 48, 32, 16)])
def test_lowrank_refactorization_equals_the_dense_svd(d_out, d_in, k, target):
    """Same truncated reconstruction and same relative error as SVD of the dense delta."""
    from amaun.adapters.delta import LayerDelta
    from amaun.fusion.refactorize import _svd_factorize, _svd_factorize_lowrank

    generator = torch.Generator().manual_seed(d_out + k)
    # Decaying spectrum, like a trained LoRA, so the truncation actually discards something.
    scales = torch.logspace(0, -2, k)
    layer = LayerDelta(A=torch.randn(k, d_in, generator=generator),
                       B=torch.randn(d_out, k, generator=generator) * scales, scaling=1.7)
    a_dense, b_dense, err_dense = _svd_factorize(layer.dense(), target)
    a_low, b_low, err_low = _svd_factorize_lowrank(layer, target)
    assert a_low.shape == a_dense.shape and b_low.shape == b_dense.shape
    scale = float(torch.linalg.norm(layer.dense()))
    assert float(torch.linalg.norm(b_low @ a_low - b_dense @ a_dense)) / scale < 1e-5
    assert err_low == pytest.approx(err_dense, abs=1e-5)
