from collections import Counter

from ai_failure_taxonomy.storage import load_gold, load_records

from .conftest import DATA, GOLD


def test_dataset_size_and_models():
    recs = load_records(DATA)
    assert 75 <= len(recs) <= 100
    assert len({r.id for r in recs}) == len(recs)
    assert len({r.source_model for r in recs}) >= 3
    assert any(r.context for r in recs) and any(r.reference for r in recs)


def test_gold_covers_every_category_and_only_test_split(taxonomy):
    recs = {r.id: r for r in load_records(DATA)}
    gold = load_gold(GOLD)
    assert set(gold) <= set(recs)
    assert all(recs[i].split == "test" for i in gold)
    assert len(gold) < len(recs)  # held-out subset, not everything
    counts = Counter(l for g in gold.values() for l in g.labels)
    for c in taxonomy.all_labels:
        assert counts[c] >= 3, c
    assert any(len(g.labels) > 1 for g in gold.values()), "expect multi-label gold examples"
    for g in gold.values():
        assert taxonomy.normalize_labels(g.labels) == g.labels
