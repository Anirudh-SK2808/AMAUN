"""M1 -- the frozen backbone.

Master-prompt constraint 4: the backbone is frozen forever and never transmitted
(``theta_{k+1} = theta_k = theta_0``).  This module makes that *verifiable* rather than
assumed, via three mechanisms:

1. :func:`freeze_backbone` sets ``requires_grad=False`` everywhere.
2. :func:`assert_backbone_frozen` is called before every optimiser step by every trainer.
3. :func:`backbone_fingerprint` hashes a deterministic sample of weights, recorded before
   and after every training phase, so a silent mutation is detectable after the fact.

Invariant I1.4 (no duplication) is enforced by :class:`BackboneCache`, a single-slot cache:
one 7B instance maximum, ever.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from amaun.config import BackboneSpec, Quantization, ResolvedConfig, RunMode
from amaun.errors import (
    BackboneDuplicationError,
    BackboneMutatedError,
    BackboneNotFrozenError,
    ConfigError,
)
from amaun.utils.logging import get_logger

log = get_logger("models.backbone")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Dimensions for the ``dummy`` tier.  These are *test fixture* values, not experiment
# hyperparameters: the tier is hard-blocked from reporting (D-001), so they do not belong
# in configs/.  They are deliberately tiny so the full pipeline runs in seconds on CPU.
DUMMY_ARCHITECTURE = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
}


@dataclass
class LoadedBackbone:
    """A frozen backbone plus everything needed to reason about it."""

    model: Any
    tokenizer: Any
    spec: BackboneSpec
    run_mode: RunMode
    device: str
    fingerprint: str
    frozen_parameter_count: int

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


# --------------------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------------------


# Module-path segments introduced purely by adapter wrappers, never by the backbone itself.
_WRAPPER_SEGMENTS = frozenset({"base_layer", "base"})


def _is_adapter_parameter(name: str) -> bool:
    """True for parameters that belong to an adapter rather than to the backbone."""
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in ("lora_a", "lora_b", "lora_embedding", "fusion_logit", "modules_to_save")
    )


def _normalise_parameter_name(name: str) -> str:
    """Strip PEFT/fusion wrapper prefixes so a name is stable across injections.

    Without this, merely attaching LoRA would change every backbone parameter's path
    (``model.layers.0...`` becomes ``base_model.model.model.layers.0...``) and the fingerprint
    would report a mutation where none occurred.
    """
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # PEFT renames a targeted module's weight to ``<path>.base_layer.weight``, and
    # FusedLoRALinear wraps the original module as ``<path>.base``.  Dropping those wrapper
    # segments makes the name identical to the un-adapted one.
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def backbone_fingerprint(model: Any, *, samples_per_tensor: int = 8) -> str:
    """Hash a deterministic sample of **backbone** weights.

    Hashing every weight of a 7B model on every phase boundary would be prohibitively
    slow, so we sample deterministically: for each parameter, in sorted normalised-name
    order, take ``samples_per_tensor`` evenly spaced elements of the flattened tensor plus
    its shape and dtype.  Any optimiser step that touched a parameter changes essentially
    all of its elements, so this detects mutation reliably while costing milliseconds.

    Adapter parameters are **excluded**, and names are normalised past PEFT/fusion wrapper
    prefixes, so the fingerprint is comparable before injection, during fusion, and after
    teardown.  It answers exactly one question: did theta move?
    """
    digest = hashlib.sha256()
    with torch.no_grad():
        entries = [
            (_normalise_parameter_name(name), param)
            for name, param in model.named_parameters()
            if not _is_adapter_parameter(name)
        ]
        for name, param in sorted(entries, key=lambda kv: kv[0]):
            digest.update(name.encode())
            digest.update(str(tuple(param.shape)).encode())
            digest.update(str(param.dtype).encode())
            flat = param.detach().reshape(-1)
            if flat.numel() == 0:
                continue
            step = max(1, flat.numel() // samples_per_tensor)
            sampled = flat[::step][:samples_per_tensor].to(torch.float32).cpu()
            digest.update(sampled.numpy().tobytes())
    return digest.hexdigest()


def freeze_backbone(model: Any) -> int:
    """Set ``requires_grad=False`` on every parameter; return how many were frozen."""
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
        count += 1
    model.eval()
    return count


def adapter_parameter_names(model: Any) -> set[str]:
    """Names of parameters that legitimately train: LoRA factors and fusion coefficients.

    We identify them by marker substrings rather than by type, because PEFT injects LoRA
    parameters under ``lora_A``/``lora_B`` names and our fusion module uses ``fusion_logit``.
    Anything else that is trainable is a bug.
    """
    return {name for name, _ in model.named_parameters() if _is_adapter_parameter(name)}


def assert_backbone_frozen(model: Any, *, allow: Iterable[str] | None = None) -> None:
    """Raise :class:`BackboneNotFrozenError` if any non-adapter parameter is trainable.

    Called before every optimiser step.  The check is cheap (a walk over parameter
    metadata, no tensor reads) and catches the classic PEFT footgun where
    ``prepare_model_for_kbit_training`` or ``enable_input_require_grads`` re-enables grads
    on backbone modules (invariant I1.3).
    """
    permitted = set(allow) if allow is not None else adapter_parameter_names(model)
    offenders = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name not in permitted
    ]
    if offenders:
        raise BackboneNotFrozenError(
            f"{len(offenders)} non-adapter parameter(s) have requires_grad=True, violating the "
            f"frozen-backbone constraint. First offenders: {offenders[:5]}"
        )


def assert_fingerprint_unchanged(model: Any, expected: str, *, phase: str = "") -> None:
    actual = backbone_fingerprint(model)
    if actual != expected:
        raise BackboneMutatedError(
            f"backbone weights changed during phase '{phase}': fingerprint {expected[:12]} -> "
            f"{actual[:12]}. The backbone must satisfy theta_{{k+1}} = theta_k = theta_0."
        )


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _build_dummy(spec: BackboneSpec, device: str) -> tuple[Any, Any]:
    """Construct a tiny randomly-initialised Qwen2 and a matching byte-level tokenizer.

    Entirely offline and in-process: no ``from_pretrained``, no network, no cache.  This is
    what lets ``pytest`` exercise the real code paths (PEFT injection, fusion, KD,
    aggregation) in seconds, which idea.md S14 requires before any expensive run.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        **DUMMY_ARCHITECTURE,
        attn_implementation=spec.attn_implementation,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(0)  # the dummy backbone must be identical across processes
    model = Qwen2ForCausalLM(config).to(_DTYPES[spec.dtype]).to(device)
    tokenizer = _DummyTokenizer(vocab_size=DUMMY_ARCHITECTURE["vocab_size"])
    return model, tokenizer


class _DummyTokenizer:
    """Deterministic byte-level tokenizer for the ``dummy`` tier.

    Implements only the surface AMAUN uses.  Byte-level means every string round-trips and
    no vocabulary file is needed, keeping the tier fully offline.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._offset = 3
        self.name_or_path = "dummy-byte-tokenizer"

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._offset + (b % (self.vocab_size - self._offset)) for b in text.encode("utf-8")]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        out = bytearray()
        for token in ids:
            token = int(token)
            if token < self._offset:
                if skip_special_tokens:
                    continue
                token = self._offset
            out.append((token - self._offset) % 256)
        return out.decode("utf-8", errors="replace")

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else text
        batch = [self.encode(t) for t in texts]
        max_len = kwargs.get("max_length")
        if kwargs.get("truncation") and max_len:
            batch = [ids[:max_len] for ids in batch]
        if kwargs.get("padding"):
            width = max_len if (kwargs.get("padding") == "max_length" and max_len) else max(len(i) for i in batch)
            batch = [ids + [self.pad_token_id] * (width - len(ids)) for ids in batch]
        result = {
            "input_ids": batch,
            "attention_mask": [[1 if t != self.pad_token_id else 0 for t in ids] for ids in batch],
        }
        if kwargs.get("return_tensors") == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        return result


def _load_pretrained(
    spec: BackboneSpec,
    device: str,
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load the pinned checkpoint from a local snapshot.

    ``cache_dir`` / ``local_files_only`` come from ``model.*`` in the resolved config and
    ``revision`` from :class:`BackboneSpec`, so the *exact* checkpoint a run used is recorded in
    ``config_used.json`` rather than being whatever ``main`` happened to point at.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec.model_id is None:
        raise ConfigError("model_id is None for a non-dummy run mode")

    kwargs: dict[str, Any] = {
        "dtype": _DTYPES[spec.dtype],
        "attn_implementation": spec.attn_implementation,
        "revision": spec.revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if spec.quantization is not Quantization.NONE:
        from transformers import BitsAndBytesConfig

        if spec.quantization is Quantization.NF4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_DTYPES[spec.dtype],
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}

    log.info(
        "loading backbone %s (revision=%s, dtype=%s, quant=%s, cache_dir=%s, local_files_only=%s)",
        spec.model_id, spec.revision, spec.dtype, spec.quantization.value,
        cache_dir, local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    if spec.quantization is Quantization.NONE:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class BackboneCache:
    """Single-slot backbone cache (invariant I1.4).

    A second *distinct* request evicts the resident instance rather than duplicating it, so
    at most one full-precision backbone is ever resident (brief S20, development rule 6).
    """

    _resident: LoadedBackbone | None = None
    _key: tuple[str, str, str] | None = None

    @classmethod
    def get(cls, cfg: ResolvedConfig, *, strict_no_duplicate: bool = False) -> LoadedBackbone:
        key = (cfg.run_mode.value, str(cfg.backbone.model_id), cfg.device)
        if cls._resident is not None and cls._key == key:
            return cls._resident
        if cls._resident is not None:
            if strict_no_duplicate:
                raise BackboneDuplicationError(
                    f"a backbone for {cls._key} is resident; refusing to load {key} as well"
                )
            log.info("evicting resident backbone %s to load %s", cls._key, key)
            cls.release()
        loaded = _load(cfg)
        cls._resident, cls._key = loaded, key
        return loaded

    @classmethod
    def release(cls) -> None:
        if cls._resident is not None:
            # The injection record is keyed by model identity; drop it with the model so a
            # freshly loaded backbone starts un-injected.
            from amaun.adapters.lora import forget_injection

            forget_injection(cls._resident.model)
        cls._resident = None
        cls._key = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def resident_count(cls) -> int:
        return 0 if cls._resident is None else 1


def _load(cfg: ResolvedConfig) -> LoadedBackbone:
    spec = cfg.backbone
    if cfg.run_mode is RunMode.DUMMY:
        model, tokenizer = _build_dummy(spec, cfg.device)
    else:
        model, tokenizer = _load_pretrained(
            spec,
            cfg.device,
            cache_dir=cfg.get("model.cache_dir", None),
            local_files_only=bool(cfg.get("model.local_files_only", False)),
        )

    frozen = freeze_backbone(model)
    assert_backbone_frozen(model, allow=set())  # nothing may be trainable yet
    fingerprint = backbone_fingerprint(model)
    log.info(
        "backbone ready: %s params frozen, fingerprint=%s, device=%s",
        frozen, fingerprint[:12], cfg.device,
    )
    return LoadedBackbone(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        run_mode=cfg.run_mode,
        device=cfg.device,
        fingerprint=fingerprint,
        frozen_parameter_count=frozen,
    )


def load_backbone(cfg: ResolvedConfig, *, use_cache: bool = True) -> LoadedBackbone:
    """Load (or reuse) the frozen backbone for this run."""
    return BackboneCache.get(cfg) if use_cache else _load(cfg)
