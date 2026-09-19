"""Shared decoding configuration -- identical across every arm (invariant I12.2).

If AMAUN sampled and a baseline used greedy decoding, part of any measured difference would
be decoding, not method.  Every arm therefore builds its decoding parameters from this one
place, and the resolved parameters are logged with the results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool
    num_beams: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "GenerationConfig":
        required = ("do_sample", "num_beams", "max_new_tokens", "temperature", "top_p", "repetition_penalty")
        missing = [key for key in required if key not in section]
        if missing:
            raise ValueError(f"generation config missing keys: {missing}")
        return cls(
            do_sample=bool(section["do_sample"]),
            num_beams=int(section["num_beams"]),
            max_new_tokens=int(section["max_new_tokens"]),
            temperature=float(section["temperature"]),
            top_p=float(section["top_p"]),
            repetition_penalty=float(section["repetition_penalty"]),
        )

    def to_kwargs(self, *, pad_token_id: int | None, eos_token_id: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            # Passing temperature/top_p with do_sample=False triggers warnings and has no
            # effect; omitting them keeps the greedy path unambiguous.
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
