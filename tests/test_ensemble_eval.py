import pytest

from ai_failure_taxonomy.classifiers.ensemble import combine, jaccard
from ai_failure_taxonomy.evaluate import evaluate
from ai_failure_taxonomy.schema import ClassifierOutput


def out(name, taxonomy, **scores):
    s = {c: scores.get(c, 0.05) for c in taxonomy.ids}
    for k, v in scores.items():
        s[k] = v
    labels = [c for c, v in s.items() if v is not None and v >= 0.5] or ["none"]
    return ClassifierOutput(name=name, scores=s, labels=labels)


def test_agreement_gives_high_confidence(taxonomy):
    e = combine(out("h", taxonomy, over_refusal=0.92), out("l", taxonomy, over_refusal=0.9), taxonomy)
    assert e.labels == ["over_refusal"]
    assert e.agreement == 1.0 and e.confidence > 0.7


def test_disagreement_lowers_confidence(taxonomy):
    agree = combine(out("h", taxonomy, formatting_failure=0.93), out("l", taxonomy, formatting_failure=0.9), taxonomy)
    disagree = combine(out("h", taxonomy, formatting_failure=0.93), out("l", taxonomy), taxonomy)
    assert disagree.confidence < agree.confidence
    assert disagree.agreement == 0.0
    assert "formatting_failure" in disagree.labels  # rule evidence for formatting is trusted


def test_asymmetric_weights(taxonomy):
    # The heuristic found no citation pattern, the LLM is confident: LLM should win for hallucination.
    e = combine(out("h", taxonomy, hallucination=0.2), out("l", taxonomy, hallucination=0.85), taxonomy)
    assert "hallucination" in e.labels


def test_abstention_uses_other_source_with_discount(taxonomy):
    h = out("h", taxonomy, unsupported_claim=None)
    l = out("l", taxonomy, unsupported_claim=0.9)
    e = combine(h, l, taxonomy)
    assert e.scores["unsupported_claim"] == pytest.approx(0.9)
    assert e.category_confidence["unsupported_claim"] == pytest.approx(0.8 * 0.8)


def test_llm_failure_falls_back_to_heuristic(taxonomy):
    failed = ClassifierOutput(name="llm", scores={c: None for c in taxonomy.ids}, labels=["none"], ok=False, error="x")
    h = out("h", taxonomy, omission=0.9)
    e = combine(h, failed, taxonomy)
    assert e.labels == ["omission"] and e.agreement == 0.0
    # single-source omission confidence |2*0.9-1|*0.8 = 0.64, then the 0.7 LLM-failure discount
    assert e.confidence == pytest.approx(0.64 * 0.7, abs=1e-3)


def test_jaccard():
    assert jaccard(["a"], ["a"]) == 1.0
    assert jaccard(["a", "b"], ["a"]) == 0.5
    assert jaccard([], []) == 1.0


def test_evaluate_hand_computed(taxonomy):
    gold = {"1": ["hallucination"], "2": ["omission", "hallucination"], "3": ["none"], "4": ["over_refusal"]}
    pred = {"1": ["hallucination"], "2": ["hallucination"], "3": ["omission"], "4": ["none"]}
    r = evaluate(gold, pred, taxonomy)
    h, o, refuse = r.per_category["hallucination"], r.per_category["omission"], r.per_category["over_refusal"]
    assert (h.tp, h.fp, h.fn, h.f1) == (2, 0, 0, 1.0)
    assert (o.tp, o.fp, o.fn) == (0, 1, 1) and o.f1 == 0.0
    assert (refuse.tp, refuse.fn) == (0, 1)
    # Micro: TP=2, FP=1, FN=2 -> P=2/3, R=1/2, F1=4/7
    assert r.micro_precision == pytest.approx(2 / 3, abs=1e-4)
    assert r.micro_recall == pytest.approx(0.5)
    assert r.micro_f1 == pytest.approx(4 / 7, abs=1e-4)
    # Macro over categories with support (hallucination, omission, over_refusal): (1 + 0 + 0) / 3
    assert r.macro_f1 == pytest.approx(1 / 3, abs=1e-4)
    assert r.exact_match == 0.25
    assert r.hamming_loss == pytest.approx(3 / (4 * 8), abs=1e-4)  # 3 wrong cells of 4 items x 8 categories
    assert r.none_accuracy == 0.5
    # Confusion: gold omission item was predicted hallucination; gold none predicted omission.
    assert r.confusion["omission"]["hallucination"] == 1
    assert r.confusion["none"]["omission"] == 1
    assert r.confusion["over_refusal"]["none"] == 1
    assert {e["id"] for e in r.errors} == {"2", "3", "4"}


def test_evaluate_requires_overlap(taxonomy):
    with pytest.raises(ValueError):
        evaluate({"a": ["none"]}, {"b": ["none"]}, taxonomy)
