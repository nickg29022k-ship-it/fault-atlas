import json

import pytest

from ai_failure_taxonomy.classifiers import predict_all
from ai_failure_taxonomy.export import ExportOptions, build_training_rows
from ai_failure_taxonomy.review import (
    apply_batch, build_queue, correction_stats, interactive_review, make_correction, resolved_labels,
)
from ai_failure_taxonomy.storage import latest_corrections, load_corrections, load_records

from .conftest import DATA, make_record


@pytest.fixture(scope="module")
def records():
    return load_records(DATA)


@pytest.fixture(scope="module")
def preds(records, taxonomy):
    return predict_all(records, taxonomy=taxonomy)


def test_queue_prioritises_low_confidence_and_disagreement(preds):
    q = build_queue(preds)
    assert len(q) == len(preds)
    prios = [i.priority for i in q]
    assert prios == sorted(prios, reverse=True)
    top, bottom = q[0].prediction, q[-1].prediction
    assert top.confidence < bottom.confidence
    assert q[0].reasons
    # Items in the top of the queue are disproportionately disagreements.
    top10 = q[:10]
    assert sum(i.prediction.agreement < 1 for i in top10) >= 7


def test_queue_skips_reviewed_items(preds, taxonomy):
    c = make_correction(preds[0], None, "r", taxonomy)
    q = build_queue(preds, [c])
    assert preds[0].id not in {i.prediction.id for i in q}
    assert len(build_queue(preds, [c], limit=5)) == 5


def test_batch_corrections_and_audit_trail(preds, taxonomy, tmp_path):
    p0, p1 = preds[0], preds[1]
    batch = tmp_path / "batch.jsonl"
    batch.write_text(
        json.dumps({"item_id": p0.id, "labels": ["hallucination", "omission"], "notes": "missed", "reviewer": "ana"}) + "\n"
        + json.dumps({"item_id": p1.id, "labels": None}) + "\n"
    )
    log = tmp_path / "corrections.jsonl"
    done = apply_batch(preds, batch, log, taxonomy, default_reviewer="bot")
    assert [c.action for c in done] == ["correct" if set(p0.final_labels) != {"hallucination", "omission"} else "accept", "accept"]
    saved = load_corrections(log)
    assert len(saved) == 2
    first = saved[0]
    assert first.original_labels == p0.final_labels
    assert first.corrected_labels == ["hallucination", "omission"]
    assert first.reviewer == "ana" and first.notes == "missed" and first.source == "batch" and first.timestamp
    assert saved[1].reviewer == "bot" and saved[1].corrected_labels == p1.final_labels

    # Append-only: a later decision on the same item supersedes the earlier one but both are kept.
    batch.write_text(json.dumps({"item_id": p0.id, "labels": ["none"], "timestamp": "2999-01-01T00:00:00+00:00"}) + "\n")
    apply_batch(preds, batch, log, taxonomy)
    saved = load_corrections(log)
    assert len(saved) == 3
    assert latest_corrections(saved)[p0.id].corrected_labels == ["none"]
    stats = correction_stats(log)
    assert stats["log_entries"] == 3 and stats["items_reviewed"] == 2


def test_batch_rejects_bad_input(preds, taxonomy, tmp_path):
    b = tmp_path / "b.jsonl"
    b.write_text(json.dumps({"item_id": "does-not-exist", "labels": ["none"]}) + "\n")
    with pytest.raises(ValueError, match="unknown item_id"):
        apply_batch(preds, b, tmp_path / "c.jsonl", taxonomy)
    b.write_text(json.dumps({"item_id": preds[0].id, "labels": ["not_a_label"]}) + "\n")
    with pytest.raises(ValueError, match="unknown labels"):
        apply_batch(preds, b, tmp_path / "c.jsonl", taxonomy)


def test_interactive_review_with_scripted_answers(preds, records, taxonomy, tmp_path):
    from io import StringIO

    from rich.console import Console

    q = build_queue(preds, limit=3)
    answers = iter(["c", "bogus_label", "omission", "note one", "a", "", "q"])
    log = tmp_path / "c.jsonl"
    done = interactive_review(q, {r.id: r for r in records}, log, "tester", taxonomy,
                              console=Console(file=StringIO()), ask=lambda *a, **k: next(answers))
    assert len(done) == 2
    assert done[0].corrected_labels == ["omission"] and done[0].notes == "note one"
    assert done[1].action == "accept"
    assert len(load_corrections(log)) == 2


def test_resolved_labels_priority(preds, taxonomy):
    c = make_correction(preds[0], ["over_refusal"], "r", taxonomy)
    gold = {preds[0].id: ["none"], preds[1].id: ["omission"]}
    res = resolved_labels(preds, [c], gold)
    assert res[preds[0].id] == (["over_refusal"], "human")
    assert res[preds[1].id] == (["omission"], "gold")
    assert res[preds[2].id][1] in ("model", "gold")


def test_export_drop_vs_keep_low_agreement(records, preds, taxonomy):
    low = [p for p in preds if p.agreement < 0.5]
    assert low, "dataset should contain low-agreement items"
    dropped, s1 = build_training_rows(records, preds, [], taxonomy, None, ExportOptions(drop_low_agreement=True))
    kept, s2 = build_training_rows(records, preds, [], taxonomy, None, ExportOptions(drop_low_agreement=False))
    assert s1.dropped_low_agreement == len(low) and len(dropped) == len(preds) - len(low)
    assert len(kept) == len(preds) and s2.kept_flagged == len(low)
    assert all("low_agreement" in r["flags"] for r in kept if r["id"] in {p.id for p in low})


def test_export_human_correction_rescues_low_agreement_item(records, preds, taxonomy):
    low = next(p for p in preds if p.agreement < 0.5)
    c = make_correction(low, ["formatting_failure"], "r", taxonomy, notes="checked")
    rows, stats = build_training_rows(records, preds, [c], taxonomy, None, ExportOptions())
    row = next(r for r in rows if r["id"] == low.id)
    assert row["labels"] == ["formatting_failure"] and row["label_source"] == "human"
    assert row["max_severity"] == "low" and row["is_failure"] and row["review_notes"] == "checked"
    assert stats.label_sources["human"] == 1


def test_export_exclude_split_and_reviewed_only(records, preds, taxonomy):
    rows, s = build_training_rows(records, preds, [], taxonomy, None, ExportOptions(exclude_splits=["test"], drop_low_agreement=False))
    test_ids = {r.id for r in records if r.split == "test"}
    assert not ({r["id"] for r in rows} & test_ids) and s.dropped_split == len(test_ids)
    c = make_correction(preds[0], None, "r", taxonomy)
    rows, _ = build_training_rows(records, preds, [c], taxonomy, None, ExportOptions(reviewed_only=True))
    assert [r["id"] for r in rows] == [preds[0].id]


def test_make_correction_detects_accept(preds, taxonomy):
    p = preds[0]
    c = make_correction(p, list(p.final_labels), "r", taxonomy)
    assert c.action == "accept"
    rec = make_record()
    assert rec.split == "dev"
