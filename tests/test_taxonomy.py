import pytest

from ai_failure_taxonomy.taxonomy import SEVERITY_ORDER

EXPECTED = {
    "hallucination", "factual_error", "omission", "reasoning_error",
    "formatting_failure", "over_refusal", "instruction_violation", "unsupported_claim",
}


def test_has_all_eight_categories(taxonomy):
    assert set(taxonomy.ids) == EXPECTED
    assert taxonomy.none_label == "none"
    assert taxonomy.all_labels[-1] == "none"


def test_every_category_is_fully_specified(taxonomy):
    for c in taxonomy.categories:
        assert c.definition.strip()
        assert c.decision_rules and c.positive_examples and c.negative_examples
        assert c.severity in SEVERITY_ORDER
        assert c.remediation.strip()
        assert taxonomy.severity_weight(c.id) == taxonomy.severity_weights[c.severity]


def test_normalize_labels_orders_dedupes_and_handles_none(taxonomy):
    assert taxonomy.normalize_labels(["omission", "hallucination", "omission"]) == ["hallucination", "omission"]
    assert taxonomy.normalize_labels([]) == ["none"]
    # A failure label wins over 'none' (none is exclusive).
    assert taxonomy.normalize_labels(["none", "over_refusal"]) == ["over_refusal"]
    with pytest.raises(ValueError):
        taxonomy.normalize_labels(["made_up_label"])


def test_custom_taxonomy_validation(tmp_path):
    from ai_failure_taxonomy.taxonomy import load_taxonomy

    bad = tmp_path / "t.yaml"
    bad.write_text(
        "version: 1\nseverity_weights: {low: 1}\ncategories:\n"
        "  - {id: a, name: A, severity: extreme, definition: d, decision_rules: [r], positive_examples: [p], negative_examples: [n]}\n"
    )
    with pytest.raises(Exception):
        load_taxonomy(bad)
