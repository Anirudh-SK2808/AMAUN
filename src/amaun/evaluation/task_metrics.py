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

BERTScore needs a second pretrained model whose identity must be held constant and reported, so
it is computed only when a run configures one (``evaluation.bertscore``,
``amaun.evaluation.bertscore``) and passes the scorer in; without one it is reported as
``bertscore_f1__unavailable`` rather than silently substituted.

``entity_f1`` (l4_full_study NER, D-021) scores ``TYPE: span`` lines by exact (type, span) match,
per sentence, averaged over sentences; ``entity_f1_micro`` is the corpus-level micro F1.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Iterable, Sequence

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


# -- named entities ------------------------------------------------------------------
NO_ENTITIES = "none"


def parse_entities(text: str) -> Counter:
    """``TYPE: span`` lines -> multiset of ``(TYPE, span)``.  ``none`` / blank lines are empty.

    Matching is exact after trimming, upper-casing the type and collapsing internal whitespace in
    the span; a line without ``:`` is not an entity.
    """
    found: Counter = Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower() == NO_ENTITIES or ":" not in line:
            continue
        kind, _, span = line.partition(":")
        kind, span = kind.strip().upper(), " ".join(span.split())
        if kind and span:
            found[(kind, span)] += 1
    return found


def entity_f1(prediction: str, reference: str) -> float:
    """Exact (type, span) F1 for one sentence; no entities on both sides scores 1."""
    predicted, referenced = parse_entities(prediction), parse_entities(reference)
    if not predicted and not referenced:
        return 1.0
    overlap = sum((predicted & referenced).values())
    return _f_measure(overlap, sum(predicted.values()), sum(referenced.values()))


def entity_f1_micro(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus micro F1 over all (type, span) entities."""
    overlap = predicted_total = reference_total = 0
    for prediction, reference in zip(predictions, references):
        predicted, referenced = parse_entities(prediction), parse_entities(reference)
        overlap += sum((predicted & referenced).values())
        predicted_total += sum(predicted.values())
        reference_total += sum(referenced.values())
    if predicted_total == 0 and reference_total == 0:
        return 1.0
    return _f_measure(overlap, predicted_total, reference_total)


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
    "entity_f1": entity_f1,
}

# Metrics computed by a configured model-based scorer (``scorers`` argument below).
SCORER_METRICS = ("bertscore_f1",)

# Metrics deliberately not implemented, with the reason.  Requesting one raises rather than
# silently substituting a different metric.
UNIMPLEMENTED_METRICS: dict[str, str] = {
    "bertscore_f1": (
        "BERTScore requires a second pretrained scorer model whose identity, layer, and "
        "rescaling baseline must be held constant across every arm and reported. Configure "
        "evaluation.bertscore so the evaluator passes a scorer (amaun.evaluation.bertscore)."
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


def compute_with_examples(
    metrics: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    mean_token_loss: float | None = None,
    scorers: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """A task's metric set plus every per-example score that has one (pairwise and scorer metrics).

    Corpus-level metrics (perplexity, distinct-2, micro entity F1) have no per-example value.
    A scorer metric with no configured scorer is reported as ``<metric>__unavailable``.
    """
    if len(predictions) != len(references):
        raise EvaluationError("prediction/reference count mismatch")
    out: dict[str, float] = {}
    per_example: dict[str, list[float]] = {}
    scorers = scorers or {}
    for metric in metrics:
        if metric == "perplexity":
            out["perplexity"] = (
                perplexity_from_loss(mean_token_loss) if mean_token_loss is not None else float("nan")
            )
        elif metric == "distinct_2":
            out["distinct_2"] = distinct_n(predictions, 2)
        elif metric == "entity_f1_micro":
            out[metric] = entity_f1_micro(predictions, references)
        elif metric in SCORER_METRICS and metric in scorers:
            values = [float(v) for v in scorers[metric].score(predictions, references)]
            per_example[metric] = values
            out[metric] = sum(values) / len(values) if values else float("nan")
        elif metric in UNIMPLEMENTED_METRICS:
            out[f"{metric}__unavailable"] = float("nan")
        else:
            if metric not in PAIRWISE_METRICS:
                compute_pairwise(metric, predictions, references)   # raises the standard error
            function = PAIRWISE_METRICS[metric]
            values = [function(p, r) for p, r in zip(predictions, references)]
            per_example[metric] = values
            out[metric] = sum(values) / len(values) if values else float("nan")
    return out, per_example


def compute_all(
    metrics: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    mean_token_loss: float | None = None,
    scorers: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute a task's declared metric set, skipping unimplemented ones with a note."""
    return compute_with_examples(metrics, predictions, references, mean_token_loss=mean_token_loss,
                                 scorers=scorers)[0]
