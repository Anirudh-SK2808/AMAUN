"""Console logging.  Result data goes to JSONL via tracking, never to the log."""

from __future__ import annotations

import logging
import sys

from amaun.config import RunMode

_CONFIGURED = False


def setup_logging(level: str = "INFO", run_mode: RunMode | None = None) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("amaun")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        prefix = ""
        if run_mode is not None and not run_mode.is_reportable:
            # idea.md S0.1.1: non-reportable output must be visibly labelled.
            prefix = f"[{run_mode.value.upper()}] "
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s {prefix}%(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"amaun.{name}")
