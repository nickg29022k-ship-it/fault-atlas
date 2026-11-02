import json

from typer.testing import CliRunner

from ai_failure_taxonomy.analysis import analyze, model_feedback, render_report
from ai_failure_taxonomy.cli import app

from .conftest import DATA, GOLD, SAMPLE_CORR, make_record


def synthetic(taxonomy):
    recs, labels = [], {}
    i = 0
    # Model B refuses 3/5 medical prompts, model A 1/5 -> 3x.
    for model, n_ref in (("A", 1), ("B", 3)):
        for k in range(5):
            i += 1
            r = make_record(id=f"m{i}", source_model=model, domain="medical")
            recs.append(r)
            labels[r.id] = (["over_refusal"] if k < n_ref else ["none"], "gold")
        for k in range(4):
            i += 1
            r = make_record(id=f"c{i}", source_model=model, domain="coding")
            recs.append(r)
            labs = ["hallucination", "unsupported_claim"] if (model == "A" and k < 2) else ["none"]
            labels[r.id] = (labs, "human")
    return recs, labels


def test_analysis_rates_cooccurrence_and_insights(taxonomy):
    recs, labels = synthetic(taxonomy)
    an = analyze(recs, labels, taxonomy)
    assert an.rates["B"]["over_refusal"] == round(3 / 9, 4)
    assert an.domain_counts["B"]["medical"]["over_refusal"] == 3
    assert an.failing_by_model_domain["A"]["coding"] == 2
    assert an.cooccurrence["hallucination"]["unsupported_claim"] == 2
    assert an.lift["hallucination"]["unsupported_claim"] > 1
    assert an.severity_counts["A"]["critical"] == 2
    assert any("B over-refuses on medical prompts at 3.0x the rate of A" in s for s in an.insights), an.insights
    assert any("co-occur" in s for s in an.insights)
    fb = model_feedback(an, taxonomy, "B")
    assert fb["failures"][0]["category"] == "over_refusal"
    assert fb["failures"][0]["domain_hotspots"][0]["domain"] == "medical"
    assert fb["failures"][0]["recommendation"]
    md = render_report(an, taxonomy)
    assert "## Key insights" in md and "| over_refusal | medium |" in md


def test_cli_end_to_end(tmp_path):
    runner = CliRunner()
    out = tmp_path / "run"
    res = runner.invoke(app, ["run-all", "--data", str(DATA), "--gold", str(GOLD),
                              "--sample-corrections", str(SAMPLE_CORR), "--out-dir", str(out)])
    assert res.exit_code == 0, res.output
    for f in ["predictions.jsonl", "corrections.jsonl", "eval.json", "eval.md", "training.jsonl",
              "report.md", "analysis.json", "review_queue.jsonl", "feedback/beta-13b.json"]:
        assert (out / f).exists(), f
    ev = json.loads((out / "eval.json").read_text())
    assert set(ev) == {"heuristic", "llm", "ensemble", "ensemble+review"}
    assert ev["ensemble+review"]["micro"]["f1"] >= ev["ensemble"]["micro"]["f1"]
    report = (out / "report.md").read_text()
    assert "over-refuses on medical prompts" in report
    rows = [json.loads(l) for l in (out / "training.jsonl").read_text().splitlines()]
    assert any(r["label_source"] == "human" for r in rows)


def test_cli_interactive_review(tmp_path):
    runner = CliRunner()
    preds = tmp_path / "p.jsonl"
    assert runner.invoke(app, ["classify", "--data", str(DATA), "--out", str(preds), "--limit", "12"]).exit_code == 0
    log = tmp_path / "c.jsonl"
    res = runner.invoke(app, ["review", "--predictions", str(preds), "--data", str(DATA), "--corrections", str(log),
                              "--reviewer", "alice", "--limit", "2"], input="c\nomission\nlooks incomplete\ns\n")
    assert res.exit_code == 0, res.output
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["reviewer"] == "alice" and rows[0]["corrected_labels"] == ["omission"]
    assert rows[0]["notes"] == "looks incomplete" and rows[0]["source"] == "interactive"


def test_cli_heuristic_only_and_bad_backend(tmp_path, monkeypatch):
    runner = CliRunner()
    p = tmp_path / "h.jsonl"
    assert runner.invoke(app, ["classify", "--data", str(DATA), "--out", str(p), "--mode", "heuristic"]).exit_code == 0
    first = json.loads(p.read_text().splitlines()[0])
    assert first["llm"] is None
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = runner.invoke(app, ["classify", "--data", str(DATA), "--out", str(p), "--backend", "anthropic", "--limit", "1"])
    assert res.exit_code == 2 and "ANTHROPIC_API_KEY" in res.output
