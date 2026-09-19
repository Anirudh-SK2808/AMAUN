"""M12 -- task metrics (brief S15).

Implemented here rather than pulled from a metrics library so that (a) the exact definition
is auditable, (b) AMAUN and every baseline provably share one implementation (invariant
I12.3), and (c) the pipeline has no hard dependency on a package that may resolve differently
on a cloud host than on the dev box.

ROUGE here is the standard F-measure over unigrams/bigrams/LCS on whitespace-and-punctuation
tokens with lowercasing.  It is **not** bit-identical to the ``rouge_score`` package
(which applies Porter stemming by default), so absolute values are not directly comparable to
published numbers -- only across arms within this project.  That limitation is stated in
``docs/experiment_protocol.md`` rather than left for a reader to discover.

BERTScore is intentionally *not* implemented: it requires a second pretrained model whose
choice would need holding constant across arms and reporting. The registry raises a clear
error rather than silently substituting a different metric.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable, Sequence

from amaun.errors import EvaluationError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles and punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


# -- QA ------------------------------------------------------------------------------
def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    referenced = normalize_answer(reference).split()
    if not predicted or not referenced:
        return float(predicted == referenced)
    common = Counter(predicted) & Counter(referenced)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(referenced)
    return 2 * precision * recall / (precision + recall)


# -- ROUGE ---------------------------------------------------------------------------
def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f_measure(overlap: int, predicted_total: int, reference_total: int) -> float:
    if overlap == 0 or predicted_total == 0 or reference_total == 0:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    predicted = tokenize(prediction)
    referenced = tokenize(reference)
    predicted_grams = _ngrams(predicted, n)
    reference_grams = _ngrams(referenced, n)
    overlap = sum((predicted_grams & reference_grams).values())
    return _f_measure(overlap, sum(predicted_grams.values()), sum(reference_grams.values()))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for l_token in left:
        current = [0]
        for index, r_token in enumerate(right):
            current.append(
                previous[index] + 1 if l_token == r_token else max(current[index], previous[index + 1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction)
    referenced = tokenize(reference)
    return _f_measure(_lcs_length(predicted, referenced), len(predicted), len(referenced))


# -- generation diversity ------------------------------------------------------------
def distinct_n(predictions: Iterable[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams (dialogue diversity)."""
    total, unique = 0, set()
    for prediction in predictions:
        grams = _ngrams(tokenize(prediction), n)
        total += sum(grams.values())
        unique.update(grams)
    return len(unique) / total if total else 0.0


def perplexity_from_loss(mean_token_loss: float) -> float:
    """``exp(mean CE)``. Guarded so a diverged run reports ``inf`` rather than overflowing."""
    if not math.isfinite(mean_token_loss):
        return float("inf")
    try:
        return math.exp(min(mean_token_loss, 70.0))
    except OverflowError:  # pragma: no cover
        return float("inf")


# -- registry ------------------------------------------------------------------------
PairMetric = Callable[[str, str], float]

PAIRWISE_METRICS: dict[str, PairMetric] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge1": lambda p, r: rouge_n(p, r, 1),
    "rouge2": lambda p, r: rouge_n(p, r, 2),
    "rougeL": rouge_l,
}

# Metrics deliberately not implemented, with the reason.  Requesting one raises rather than
# silently substituting a different metric.
UNIMPLEMENTED_METRICS: dict[str, str] = {
    "bertscore_f1": (
        "BERTScore requires a second pretrained scorer model whose identity, layer, and "
        "rescaling baseline must be held constant across every arm and reported. Add it as an "
        "explicit experimental variable (configs/) before enabling."
    ),
    "llm_judge": (
        "LLM-as-judge requires a declared judge model, prompt, and calibration protocol, plus "
        "a bias analysis. Brief S15 lists it as optional; it is not implemented."
    ),
}


def compute_pairwise(metric: str, predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean of a pairwise metric over a prediction/reference set."""
    if metric in UNIMPLEMENTED_METRICS:
        raise EvaluationError(f"metric '{metric}' is not implemented: {UNIMPLEMENTED_METRICS[metric]}")
    function = PAIRWISE_METRICS.get(metric)
    if function is None:
        raise EvaluationError(
            f"unknown metric '{metric}'; available: {sorted(PAIRWISE_METRICS)} "
            f"(unimplemented: {sorted(UNIMPLEMENTED_METRICS)})"
        )
    if len(predictions) != len(references):
        raise EvaluationError("prediction/reference count mismatch")
    if not predictions:
        return float("nan")
    return sum(function(p, r) for p, r in zip(predictions, references)) / len(predictions)


def compute_all(
    metrics: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    mean_token_loss: float | None = None,
) -> dict[str, float]:
    """Compute a task's declared metric set, skipping unimplemented ones with a note."""
    out: dict[str, float] = {}
    for metric in metrics:
        if metric == "perplexity":
            out["perplexity"] = (
                perplexity_from_loss(mean_token_loss) if mean_token_loss is not None else float("nan")
            )
        elif metric == "distinct_2":
            out["distinct_2"] = distinct_n(predictions, 2)
        elif metric in UNIMPLEMENTED_METRICS:
            out[f"{metric}__unavailable"] = float("nan")
        else:
            out[metric] = compute_pairwise(metric, predictions, references)
    return out
