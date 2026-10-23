#!/usr/bin/env python
"""Cache the BERTScore encoder named in ``evaluation.bertscore`` and verify it loads offline.

Run once on the run host before ``scripts/run_full_study.py`` (whose config sets
``local_files_only: true``, so a missing model fails the preflight instead of downloading
mid-run).  The pinned revision is checked: a different commit would change every BERTScore.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.errors import ConfigError
from amaun.evaluation.bertscore import BERTScoreConfig, BERTScorer
from amaun.experiment import add_common_arguments, resolve_from_args


def main() -> int:
    args = add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = resolve_from_args(args)
    section = cfg.get("evaluation.bertscore")
    config = BERTScoreConfig.from_config(section)
    if config.model_id is None:
        raise ConfigError("evaluation.bertscore.model_id is null in this run mode (dummy encoder); nothing to fetch")
    from huggingface_hub import snapshot_download

    path = snapshot_download(config.model_id, revision=config.revision, cache_dir=config.cache_dir)
    offline = BERTScoreConfig(**{**config.__dict__, "local_files_only": True})
    scorer = BERTScorer(offline, device="cpu").load()
    score = scorer.score(["the cat sat on the mat"], ["a cat was sitting on the mat"])[0]
    print(f"cached {config.model_id}@{config.revision} at {path}; offline load ok; sanity BERTScore {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
