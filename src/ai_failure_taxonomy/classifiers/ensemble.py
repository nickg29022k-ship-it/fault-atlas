"""Ensemble of the heuristic and LLM classifiers with a confidence score.

For each category c, with heuristic score h and LLM score l (either may be
None when that classifier abstained):

    s_c = w * h + (1 - w) * l      if both present
    s_c = h or l                   if only one present (confidence is discounted)

The heuristic weight w depends on the category *and* on whether the rule
fired. Rules are precise but have limited recall: when a refusal phrase or a
broken JSON payload is detected that is strong evidence (w_pos), but when no
pattern is found the rule says little about failures it cannot see (w_neg).
For example a fabricated citation pattern is reliable evidence of
hallucination, while the absence of one is weak evidence of no hallucination.

Category confidence is the distance from the decision threshold,
|2 * s_c - 1|, and item confidence is the minimum over categories, so a single
borderline category makes the whole item low-confidence. Disagreement between
the two classifiers pushes s_c toward 0.5 and lowers confidence.
"""

from __future__ import annotations

from ..schema import ClassifierOutput, EnsembleOutput, Prediction, Record
from ..taxonomy import Taxonomy, load_taxonomy
from .heuristic import HeuristicClassifier
from .llm_classifier import LLMClassifier

# category: (w_pos, w_neg) = trust in the heuristic when it fires / when it does not
HEURISTIC_WEIGHTS: dict[str, tuple[float, float]] = {
    "formatting_failure": (0.75, 0.6),
    "instruction_violation": (0.75, 0.5),
    "omission": (0.65, 0.5),
    "over_refusal": (0.7, 0.3),
    "unsupported_claim": (0.6, 0.4),
    "reasoning_error": (0.75, 0.3),
    "factual_error": (0.6, 0.2),
    "hallucination": (0.7, 0.15),
}
SINGLE_SOURCE_DISCOUNT = 0.8
LLM_FAILURE_DISCOUNT = 0.7


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def combine(
    heur: ClassifierOutput,
    llm: ClassifierOutput | None,
    taxonomy: Taxonomy,
    weights: dict[str, tuple[float, float]] | None = None,
    threshold: float = 0.5,
) -> EnsembleOutput:
    weights = {**HEURISTIC_WEIGHTS, **(weights or {})}
    scores: dict[str, float] = {}
    cat_conf: dict[str, float] = {}
    for c in taxonomy.ids:
        h = heur.scores.get(c)
        l = llm.scores.get(c) if (llm is not None and llm.ok) else None
        if h is not None and l is not None:
            w_pos, w_neg = weights.get(c, (0.5, 0.5))
            w = w_pos if h >= threshold else w_neg
            s = w * h + (1 - w) * l
            conf = abs(2 * s - 1)
        elif h is not None or l is not None:
            s = h if h is not None else l  # type: ignore[assignment]
            conf = abs(2 * s - 1) * SINGLE_SOURCE_DISCOUNT
        else:
            s, conf = 0.0, 0.5
        scores[c] = round(float(s), 4)
        cat_conf[c] = round(float(conf), 4)
    labels = [c for c in taxonomy.ids if scores[c] >= threshold] or [taxonomy.none_label]
    item_conf = min(cat_conf.values())
    if llm is not None and not llm.ok:
        item_conf *= LLM_FAILURE_DISCOUNT
    agreement = jaccard(heur.labels, llm.labels) if (llm is not None and llm.ok) else 0.0
    return EnsembleOutput(
        scores=scores,
        labels=labels,
        confidence=round(item_conf, 4),
        category_confidence=cat_conf,
        agreement=round(agreement, 4),
    )


class EnsembleClassifier:
    name = "ensemble"

    def __init__(self, backend="mock", taxonomy: Taxonomy | None = None, weights: dict[str, tuple[float, float]] | None = None):
        self.taxonomy = taxonomy or load_taxonomy()
        self.heuristic = HeuristicClassifier(self.taxonomy)
        self.llm = LLMClassifier(backend, self.taxonomy)
        self.weights = weights

    def predict(self, rec: Record) -> Prediction:
        h = self.heuristic.classify(rec)
        l = self.llm.classify(rec)
        ens = combine(h, l, self.taxonomy, self.weights)
        return Prediction(id=rec.id, source_model=rec.source_model, domain=rec.domain, heuristic=h, llm=l, ensemble=ens)


def predict_all(records: list[Record], mode: str = "ensemble", backend="mock", taxonomy: Taxonomy | None = None) -> list[Prediction]:
    """mode: `heuristic` (rules only, no LLM calls) or `ensemble` (rules + LLM + combination).

    Ensemble predictions keep both component outputs, so a single predictions
    file can be used to evaluate the heuristic, LLM and ensemble classifiers.
    """
    if mode not in ("heuristic", "ensemble"):
        raise ValueError("mode must be 'heuristic' or 'ensemble'")
    taxonomy = taxonomy or load_taxonomy()
    heur = HeuristicClassifier(taxonomy)
    llm = LLMClassifier(backend, taxonomy) if mode == "ensemble" else None
    out = []
    for rec in records:
        h = heur.classify(rec)
        l = llm.classify(rec) if llm else None
        ens = combine(h, l, taxonomy) if l is not None else None
        out.append(Prediction(id=rec.id, source_model=rec.source_model, domain=rec.domain, heuristic=h, llm=l, ensemble=ens))
    return out
