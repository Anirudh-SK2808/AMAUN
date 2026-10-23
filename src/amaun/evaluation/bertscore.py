"""BERTScore-F1 (Zhang et al. 2020), re-implemented to match ``bert_score`` 0.3.13's defaults.

``bert_score`` itself is not a dependency: its pinned release predates the ``transformers`` this
repo runs on.  What it computes is small and is reproduced here exactly:

1. encode each sentence with the scorer's tokenizer (special tokens added; RoBERTa with a prefix
   space), truncated to the model's maximum length;
2. take the hidden states of layer ``num_layers`` (bert_score truncates the encoder to that many
   layers and reads its output -- the same tensor);
3. L2-normalise every token embedding; drop the special tokens (bert_score gives CLS / SEP an idf
   weight of 0, and every other token weight 1 when ``idf=False``);
4. greedy matching: ``P`` = mean over candidate tokens of the max cosine to any reference token,
   ``R`` the converse, ``F = 2PR / (P + R)``;
5. with ``rescale_with_baseline``: ``(F - b) / (1 - b)``, ``b`` from bert_score's
   ``rescale_baseline/en/<model>.tsv`` at that layer.  An empty candidate or reference scores a raw
   0 before rescaling, as in bert_score.

The model, revision, layer and baseline are config values (``evaluation.bertscore``), recorded in
``config_used.json`` and held constant across every option of a task (the reason D-009 kept this
metric out until it was configured).  The ``dummy`` tier builds a tiny random encoder in-process,
so tests stay offline; its scores are a wiring check, never a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from amaun.errors import EvaluationError
from amaun.utils.logging import get_logger

log = get_logger("evaluation.bertscore")


@dataclass(frozen=True)
class BERTScoreConfig:
    model_id: str | None            # None => the dummy-tier random encoder
    revision: str | None
    num_layers: int
    rescale_with_baseline: bool
    baseline_f: float | None
    batch_size: int
    max_length: int
    cache_dir: str | None = None
    local_files_only: bool = False

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "BERTScoreConfig":
        required = ("num_layers", "rescale_with_baseline", "batch_size", "max_length")
        missing = [key for key in required if key not in section]
        if missing:
            raise EvaluationError(f"evaluation.bertscore missing keys: {missing}")
        rescale = bool(section["rescale_with_baseline"])
        baseline = section.get("baseline_f")
        if rescale and baseline is None:
            raise EvaluationError("evaluation.bertscore.rescale_with_baseline needs baseline_f "
                                  "(bert_score's rescale_baseline value for this model and layer)")
        return cls(
            model_id=section.get("model_id"), revision=section.get("revision"),
            num_layers=int(section["num_layers"]), rescale_with_baseline=rescale,
            baseline_f=None if baseline is None else float(baseline),
            batch_size=int(section["batch_size"]), max_length=int(section["max_length"]),
            cache_dir=section.get("cache_dir"), local_files_only=bool(section.get("local_files_only", False)),
        )


class _DummyEncoderTokenizer:
    """Byte tokenizer with CLS / SEP, for the offline random encoder."""

    cls_token_id, sep_token_id, pad_token_id = 0, 2, 1

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, *, max_length: int) -> list[int]:
        body = [3 + (b % (self.vocab_size - 3)) for b in text.encode("utf-8")][: max_length - 2]
        return [self.cls_token_id] + body + [self.sep_token_id]


class BERTScorer:
    """Scores (candidate, reference) pairs; holds one encoder for the whole run."""

    def __init__(self, config: BERTScoreConfig, *, device: str) -> None:
        self.config = config
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    # -- loading -----------------------------------------------------------------
    def load(self) -> "BERTScorer":
        if self._model is not None:
            return self
        cfg = self.config
        if cfg.model_id is None:
            from transformers import RobertaConfig, RobertaModel

            torch.manual_seed(0)
            model_cfg = RobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=max(2, cfg.num_layers),
                                      num_attention_heads=2, intermediate_size=64,
                                      max_position_embeddings=cfg.max_length + 4, pad_token_id=1)
            self._model = RobertaModel(model_cfg)
            self._tokenizer = _DummyEncoderTokenizer(300)
        else:
            from transformers import AutoModel, AutoTokenizer

            kwargs = {"revision": cfg.revision, "cache_dir": cfg.cache_dir,
                      "local_files_only": cfg.local_files_only}
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, add_prefix_space=True, **kwargs)
            self._model = AutoModel.from_pretrained(cfg.model_id, **kwargs)
        n_layers = self._model.config.num_hidden_layers
        if not 0 < cfg.num_layers <= n_layers:
            raise EvaluationError(f"bertscore num_layers={cfg.num_layers} outside 1..{n_layers}")
        # Kept on CPU between scoring calls: resident on the GPU, roberta-large (fp32, ~1.3 GiB)
        # would sit on top of the run's fusion peak (FinQA: 20.4 GiB of the L4's 22.5).
        self._model.to("cpu").eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        log.info("BERTScore encoder %s@%s layer %d, computed on %s (held on cpu between calls)",
                 cfg.model_id or "dummy", cfg.revision, cfg.num_layers, self.device)
        return self

    def release(self) -> None:
        self._model = self._tokenizer = None

    # -- scoring -----------------------------------------------------------------
    def _encode(self, text: str) -> list[int]:
        text = text.strip()
        if isinstance(self._tokenizer, _DummyEncoderTokenizer):
            return self._tokenizer.encode(text, max_length=self.config.max_length)
        return self._tokenizer.encode(text, add_special_tokens=True, max_length=self.config.max_length,
                                      truncation=True)

    def _special_ids(self) -> set[int]:
        tok = self._tokenizer
        return {i for i in (getattr(tok, "cls_token_id", None), getattr(tok, "sep_token_id", None),
                            getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None))
                if i is not None}

    @torch.no_grad()
    def _embed(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """Normalised layer-``num_layers`` embeddings of each text's non-special tokens."""
        special = self._special_ids()
        pad = getattr(self._tokenizer, "pad_token_id", 0) or 0
        out: list[torch.Tensor] = []
        for start in range(0, len(texts), self.config.batch_size):
            encoded = [self._encode(t) for t in texts[start : start + self.config.batch_size]]
            width = max(len(ids) for ids in encoded)
            ids = torch.full((len(encoded), width), pad, dtype=torch.long)
            mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for row, seq in enumerate(encoded):
                ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[row, : len(seq)] = 1
            hidden = self._model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                 output_hidden_states=True).hidden_states[self.config.num_layers]
            for row, seq in enumerate(encoded):
                keep = [i for i, token in enumerate(seq) if token not in special]
                vectors = hidden[row, keep].float()
                out.append(torch.nn.functional.normalize(vectors, dim=-1).cpu() if keep else vectors.cpu())
        return out

    def score(self, candidates: Sequence[str], references: Sequence[str]) -> list[float]:
        """Per-pair F1 (rescaled when configured)."""
        if len(candidates) != len(references):
            raise EvaluationError("bertscore: candidate/reference count mismatch")
        if not candidates:
            return []
        self.load()
        self._model.to(self.device)
        try:
            cand_emb = self._embed(list(candidates))
            ref_emb = self._embed(list(references))
        finally:
            self._model.to("cpu")
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        scores = []
        for c, r in zip(cand_emb, ref_emb):
            if c.shape[0] == 0 or r.shape[0] == 0:
                f1 = 0.0
            else:
                sim = c @ r.T
                precision = float(sim.max(dim=1).values.mean())
                recall = float(sim.max(dim=0).values.mean())
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            if self.config.rescale_with_baseline:
                f1 = (f1 - self.config.baseline_f) / (1.0 - self.config.baseline_f)
            scores.append(f1)
        return scores
