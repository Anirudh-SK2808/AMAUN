"""The decision study's preflight measures the batches training will actually allocate.

On the L4 the preflight passed with FinQA's longest-by-characters rows (774 tokens) while a
shuffled training batch reached 884 tokens and ran out of memory: middle truncation caps long
prompts, so characters are not a proxy for tokens.  It also never applied
``training.gradient_checkpointing``, so it could not check that setting either.
"""

import copy
import dataclasses

import pytest

from amaun.config import load_config, resolve
from amaun.continual.decision_study import DecisionStudy, worst_case_examples
from amaun.data.preprocess import tokenize_example
from amaun.data.splits import Example, SplitName, trainer_guard
from amaun.experiment import prepare_data
from amaun.models.backbone import DUMMY_ARCHITECTURE, BackboneCache, _DummyTokenizer, load_backbone
from amaun.utils.tracking import ExperimentTracker

from conftest import ROOT

CONFIG = ROOT / "configs" / "experiments" / "l4_decision_study.yaml"


def _tokens(tokenizer, task, example) -> int:
    return len(tokenize_example(tokenizer, example, max_input_tokens=task.max_input_tokens,
                                max_target_tokens=task.max_target_tokens,
                                prompt_truncation=task.prompt_truncation)[0])


def test_worst_case_is_chosen_by_tokens_not_characters(tasks):
    tokenizer = _DummyTokenizer(vocab_size=DUMMY_ARCHITECTURE["vocab_size"])
    task = dataclasses.replace(tasks[0], max_input_tokens=20, max_target_tokens=30, prompt_truncation="middle")
    long_prompt = Example(prompt="x" * 500, target="ab", task_id=task.task_id)      # truncated to 20
    long_target = Example(prompt="short q", target="y" * 25, task_id=task.task_id)
    examples = [long_prompt, long_target]
    # The previous selection (characters) picks the row that is shorter in tokens.
    by_characters = max(examples, key=lambda e: len(e.prompt) + len(e.target))
    assert by_characters is long_prompt
    assert _tokens(tokenizer, task, long_target) > _tokens(tokenizer, task, long_prompt)

    longest, longest_prompt = worst_case_examples(tokenizer, task, examples, 1)
    assert longest == [long_target]
    assert longest_prompt is long_prompt


@pytest.mark.parametrize("checkpointing", [True, False])
def test_preflight_measures_the_token_longest_batch_and_honours_checkpointing(tmp_path, checkpointing):
    raw = copy.deepcopy(load_config(CONFIG, config_root=ROOT / "configs"))
    raw["training"]["gradient_checkpointing"] = checkpointing
    cfg = resolve(raw, run_mode="dummy", experiment_id="pytest-preflight")
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    data = prepare_data(cfg)
    BackboneCache.release()
    backbone = load_backbone(cfg)
    try:
        study = DecisionStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker)
        enabled = []
        original = study.peft_model.gradient_checkpointing_enable

        def spy(*args, **kwargs):
            enabled.append(True)
            return original(*args, **kwargs)

        study.peft_model.gradient_checkpointing_enable = spy
        report = study.preflight()
    finally:
        BackboneCache.release()

    assert report["ok"] and report["gradient_checkpointing"] is checkpointing
    train_checks = [c for c in report["checks"] if c["check"] == "train_step"]
    assert len(enabled) == (len(train_checks) if checkpointing else 0)
    assert not getattr(backbone.model, "is_gradient_checkpointing", False), "checkpointing left enabled"
    for check in train_checks:
        task = study.tasks[check["task_id"]]
        train = trainer_guard(data.splits[task.task_id]).get(SplitName.TRAIN)
        assert check["sequence_length"] == max(_tokens(backbone.tokenizer, task, e) for e in train)
