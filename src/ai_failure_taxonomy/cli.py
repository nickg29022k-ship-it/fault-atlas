"""Command-line interface (`aft`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import analysis as an_mod
from .classifiers import predict_all
from .evaluate import evaluate as run_eval
from .evaluate import format_eval_markdown
from .export import ExportOptions, export_training
from .llm import BackendUnavailable
from .review import apply_batch, build_queue, correction_stats, interactive_review, resolved_labels
from .storage import load_corrections, load_gold, load_predictions, load_records, write_jsonl
from .taxonomy import load_taxonomy

app = typer.Typer(add_completion=False, help="Classify, review, export and analyse failures in AI-generated responses.")
console = Console()
STATE: dict = {"taxonomy": None}

DATA = Path("data/records.jsonl")
GOLD = Path("data/gold_labels.jsonl")
RUNS = Path("runs")


def tax():
    return load_taxonomy(STATE["taxonomy"])


@app.callback()
def main(taxonomy: Optional[Path] = typer.Option(None, "--taxonomy", help="Path to a taxonomy YAML (default: bundled).")):
    STATE["taxonomy"] = taxonomy


@app.command("taxonomy")
def show_taxonomy():
    """Print the taxonomy."""
    t = tax()
    table = Table("id", "severity", "definition", show_lines=True)
    for c in t.categories:
        table.add_row(c.id, c.severity, " ".join(c.definition.split()))
    console.print(table)
    console.print(f"plus '{t.none_label}' for acceptable responses (exclusive with all categories)")


@app.command()
def classify(
    data: Path = typer.Option(DATA, help="Records JSONL."),
    out: Path = typer.Option(RUNS / "predictions.jsonl", help="Where to write predictions."),
    mode: str = typer.Option("ensemble", help="heuristic | ensemble"),
    backend: str = typer.Option("mock", help="LLM backend: mock | ollama | anthropic | openai"),
    limit: Optional[int] = typer.Option(None, help="Only classify the first N records."),
):
    """Run the classifiers over a dataset."""
    records = load_records(data)[:limit] if limit else load_records(data)
    try:
        preds = predict_all(records, mode=mode, backend=backend, taxonomy=tax())
    except BackendUnavailable as e:
        console.print(f"[red]backend error:[/] {e}")
        raise typer.Exit(2)
    write_jsonl(out, preds)
    n_fail = sum(p.final_labels != [tax().none_label] for p in preds)
    n_parse = sum(1 for p in preds if p.llm is not None and not p.llm.ok)
    console.print(f"classified {len(preds)} records -> {out} ({n_fail} with failures, {n_parse} unparseable LLM outputs)")


def _corrected_predictions(preds, corrections):
    from .storage import latest_corrections

    latest = latest_corrections(corrections)
    return {p.id: (latest[p.id].corrected_labels if p.id in latest else p.final_labels) for p in preds}


@app.command()
def evaluate(
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    gold: Path = typer.Option(GOLD),
    corrections: Optional[Path] = typer.Option(None, help="If given, also score labels after merging human corrections."),
    out_json: Optional[Path] = typer.Option(RUNS / "eval.json"),
    out_md: Optional[Path] = typer.Option(RUNS / "eval.md"),
):
    """Score heuristic, LLM and ensemble predictions against gold labels."""
    t = tax()
    preds = load_predictions(predictions)
    g = {k: v.labels for k, v in load_gold(gold).items()}
    systems = {"heuristic": {p.id: p.heuristic.labels for p in preds}}
    if any(p.llm for p in preds):
        systems["llm"] = {p.id: p.llm.labels for p in preds if p.llm}
        systems["ensemble"] = {p.id: p.final_labels for p in preds}
    if corrections:
        systems["ensemble+review"] = _corrected_predictions(preds, load_corrections(corrections))
    results = {name: run_eval(g, sysp, t) for name, sysp in systems.items()}

    table = Table("classifier", "n", "macro F1", "micro F1", "exact match", "hamming")
    for name, r in results.items():
        table.add_row(name, str(r.n_items), f"{r.macro_f1:.3f}", f"{r.micro_f1:.3f}", f"{r.exact_match:.3f}", f"{r.hamming_loss:.3f}")
    console.print(table)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Classifier evaluation\n\n" + format_eval_markdown(results, t))
    console.print(f"wrote {out_json} and {out_md}")


@app.command()
def queue(
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    corrections: Path = typer.Option(RUNS / "corrections.jsonl"),
    limit: int = typer.Option(15),
    out: Optional[Path] = typer.Option(None, help="Optionally write the queue as JSONL."),
):
    """Show the review queue: low confidence and classifier disagreement first."""
    q = build_queue(load_predictions(predictions), load_corrections(corrections), limit=limit)
    table = Table("rank", "id", "model", "priority", "conf", "agree", "ensemble labels", "reasons")
    for i, item in enumerate(q, 1):
        p = item.prediction
        table.add_row(str(i), p.id, p.source_model, f"{item.priority:.2f}", f"{p.confidence:.2f}", f"{p.agreement:.2f}",
                      ", ".join(p.final_labels), "; ".join(item.reasons)[:80])
    console.print(table)
    if out:
        write_jsonl(out, [{"rank": i, "id": it.prediction.id, "priority": it.priority, "reasons": it.reasons,
                           "labels": it.prediction.final_labels} for i, it in enumerate(q, 1)])


@app.command()
def review(
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    data: Path = typer.Option(DATA),
    corrections: Path = typer.Option(RUNS / "corrections.jsonl", help="Append-only audit log of decisions."),
    reviewer: str = typer.Option("reviewer", help="Reviewer id recorded in the audit trail."),
    from_file: Optional[Path] = typer.Option(None, "--from-file", help="Apply decisions from JSONL (non-interactive)."),
    limit: int = typer.Option(10, help="Interactive mode: number of queue items to review."),
):
    """Review predictions interactively, or apply corrections from a JSONL file."""
    t = tax()
    preds = load_predictions(predictions)
    if from_file:
        try:
            done = apply_batch(preds, from_file, corrections, t, default_reviewer=reviewer)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
    else:
        q = build_queue(preds, load_corrections(corrections), limit=limit)
        if not q:
            console.print("review queue is empty")
            return
        recs = {r.id: r for r in load_records(data)}
        done = interactive_review(q, recs, corrections, reviewer, t, console=console)
    n_corr = sum(c.action == "correct" for c in done)
    console.print(f"recorded {len(done)} decisions ({n_corr} corrections, {len(done) - n_corr} accepts) -> {corrections}")


@app.command("export")
def export_cmd(
    data: Path = typer.Option(DATA),
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    corrections: Path = typer.Option(RUNS / "corrections.jsonl"),
    gold: Optional[Path] = typer.Option(GOLD),
    out: Path = typer.Option(RUNS / "training.jsonl"),
    min_agreement: float = typer.Option(0.5, help="Heuristic/LLM Jaccard below this counts as low agreement."),
    min_confidence: float = typer.Option(0.0, help="Drop unreviewed items with ensemble confidence below this."),
    drop_low_agreement: bool = typer.Option(True, "--drop-low-agreement/--keep-low-agreement",
                                            help="Drop unreviewed low-agreement items, or keep them with a flag."),
    use_gold: bool = typer.Option(True, "--use-gold/--no-gold", help="Use gold labels where no human correction exists."),
    exclude_split: list[str] = typer.Option([], help="Exclude records in this split (repeatable), e.g. test."),
    reviewed_only: bool = typer.Option(False, help="Only export human-reviewed items."),
):
    """Export a cleaned training set (JSONL) with human corrections merged in."""
    t = tax()
    g = {k: v.labels for k, v in load_gold(gold).items()} if gold and gold.exists() else None
    opts = ExportOptions(min_agreement=min_agreement, min_confidence=min_confidence, drop_low_agreement=drop_low_agreement,
                         use_gold=use_gold, exclude_splits=list(exclude_split), reviewed_only=reviewed_only)
    stats = export_training(out, load_records(data), load_predictions(predictions), load_corrections(corrections), t, g, opts)
    console.print(f"wrote {stats.written}/{stats.total} items -> {out}")
    console.print(json.dumps(vars(stats), indent=2))


@app.command()
def report(
    data: Path = typer.Option(DATA),
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    corrections: Path = typer.Option(RUNS / "corrections.jsonl"),
    gold: Optional[Path] = typer.Option(GOLD),
    eval_md: Optional[Path] = typer.Option(None, help="Embed an evaluation markdown file in the report."),
    out_dir: Path = typer.Option(RUNS),
):
    """Aggregate failure analysis: Markdown report, analysis JSON, per-model feedback JSON."""
    t = tax()
    recs = load_records(data)
    preds = load_predictions(predictions)
    corrs = load_corrections(corrections)
    g = {k: v.labels for k, v in load_gold(gold).items()} if gold and gold.exists() else None
    labels = resolved_labels(preds, corrs, g)
    an = an_mod.analyze(recs, labels, t, preds)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = correction_stats(corrections) if corrs else None
    md = an_mod.render_report(an, t, eval_md.read_text().split("\n", 2)[2] if eval_md and eval_md.exists() else None, stats)
    (out_dir / "report.md").write_text(md)
    (out_dir / "analysis.json").write_text(json.dumps(an.to_dict(), indent=2))
    fb_dir = out_dir / "feedback"
    fb_dir.mkdir(exist_ok=True)
    for m in an.models:
        (fb_dir / f"{m}.json").write_text(json.dumps(an_mod.model_feedback(an, t, m), indent=2))
    console.print(f"wrote {out_dir / 'report.md'}, {out_dir / 'analysis.json'} and {len(an.models)} feedback files")
    for i in an.insights:
        console.print(f"- {i}")


@app.command("run-all")
def run_all(
    data: Path = typer.Option(DATA),
    gold: Path = typer.Option(GOLD),
    sample_corrections: Optional[Path] = typer.Option(Path("examples/sample_corrections.jsonl"),
                                                      help="Reviewer decisions applied in batch mode."),
    backend: str = typer.Option("mock"),
    out_dir: Path = typer.Option(RUNS),
):
    """Run the full pipeline: classify, evaluate, queue, apply corrections, re-evaluate, export, report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_p, corr_p = out_dir / "predictions.jsonl", out_dir / "corrections.jsonl"
    if corr_p.exists():
        corr_p.unlink()  # fresh audit log for a reproducible run
    classify(data=data, out=preds_p, mode="ensemble", backend=backend, limit=None)
    queue(predictions=preds_p, corrections=corr_p, limit=15, out=out_dir / "review_queue.jsonl")
    if sample_corrections and sample_corrections.exists():
        review(predictions=preds_p, data=data, corrections=corr_p, reviewer="batch", from_file=sample_corrections, limit=0)
    evaluate(predictions=preds_p, gold=gold, corrections=corr_p if corr_p.exists() else None,
             out_json=out_dir / "eval.json", out_md=out_dir / "eval.md")
    export_cmd(data=data, predictions=preds_p, corrections=corr_p, gold=gold, out=out_dir / "training.jsonl",
               min_agreement=0.5, min_confidence=0.0, drop_low_agreement=True, use_gold=True, exclude_split=[],
               reviewed_only=False)
    report(data=data, predictions=preds_p, corrections=corr_p, gold=gold, eval_md=out_dir / "eval.md", out_dir=out_dir)


if __name__ == "__main__":
    app()
