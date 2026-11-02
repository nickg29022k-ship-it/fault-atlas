"""Human-in-the-loop review: prioritised queue, corrections and audit trail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .schema import Correction, Prediction, Record
from .storage import append_jsonl, latest_corrections, load_corrections, read_jsonl
from .taxonomy import Taxonomy

DISAGREEMENT_WEIGHT = 0.5
PARSE_FAILURE_BONUS = 0.25


@dataclass
class QueueItem:
    prediction: Prediction
    priority: float
    reasons: list[str]


def review_priority(pred: Prediction) -> tuple[float, list[str]]:
    """Priority = (1 - confidence) + 0.5 * (1 - agreement) [+ 0.25 if the LLM output failed to parse]."""
    reasons = []
    conf = pred.confidence
    agree = pred.agreement
    prio = 1.0 - conf
    if conf < 0.5:
        reasons.append(f"low confidence ({conf:.2f})")
    if pred.llm is not None and pred.llm.ok:
        prio += DISAGREEMENT_WEIGHT * (1.0 - agree)
        if agree < 1.0:
            reasons.append(f"classifiers disagree (heuristic={pred.heuristic.labels}, llm={pred.llm.labels})")
    elif pred.llm is not None:
        prio += PARSE_FAILURE_BONUS
        reasons.append(f"LLM output unparseable: {pred.llm.error}")
    return round(prio, 4), reasons


def build_queue(
    predictions: list[Prediction],
    corrections: list[Correction] | None = None,
    limit: int | None = None,
    min_priority: float = 0.0,
    include_reviewed: bool = False,
) -> list[QueueItem]:
    reviewed = set(latest_corrections(corrections or []))
    items = []
    for p in predictions:
        if p.id in reviewed and not include_reviewed:
            continue
        prio, reasons = review_priority(p)
        if prio < min_priority:
            continue
        items.append(QueueItem(p, prio, reasons))
    items.sort(key=lambda q: (-q.priority, q.prediction.id))
    return items[:limit] if limit else items


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_correction(
    pred: Prediction,
    labels: list[str] | None,
    reviewer: str,
    taxonomy: Taxonomy,
    notes: str = "",
    source: str = "interactive",
    timestamp: str | None = None,
) -> Correction:
    original = pred.final_labels
    if labels is None:
        corrected, action = list(original), "accept"
    else:
        corrected = taxonomy.normalize_labels(labels)
        action = "accept" if set(corrected) == set(original) else "correct"
    return Correction(
        item_id=pred.id,
        action=action,
        original_labels=original,
        corrected_labels=corrected,
        reviewer=reviewer,
        timestamp=timestamp or _now(),
        notes=notes,
        source=source,  # type: ignore[arg-type]
    )


def apply_batch(
    predictions: list[Prediction],
    batch_path: str | Path,
    corrections_path: str | Path,
    taxonomy: Taxonomy,
    default_reviewer: str = "batch",
) -> list[Correction]:
    """Apply reviewer decisions from a JSONL file and append them to the audit log.

    Each input line: {"item_id": "...", "labels": [...] | null, "notes": "...", "reviewer": "..."}.
    Omitting `labels` (or setting it to null) accepts the model's labels.
    """
    by_id = {p.id: p for p in predictions}
    out = []
    for lineno, row in enumerate(read_jsonl(batch_path), 1):
        item_id = row.get("item_id") or row.get("id")
        if item_id not in by_id:
            raise ValueError(f"{batch_path}:{lineno}: unknown item_id {item_id!r}")
        try:
            corr = make_correction(
                by_id[item_id],
                row.get("labels"),
                row.get("reviewer") or default_reviewer,
                taxonomy,
                notes=row.get("notes", ""),
                source="batch",
                timestamp=row.get("timestamp"),
            )
        except ValueError as e:
            raise ValueError(f"{batch_path}:{lineno}: {e}") from e
        out.append(corr)
    append_jsonl(corrections_path, out)
    return out


def interactive_review(
    queue: list[QueueItem],
    records: dict[str, Record],
    corrections_path: str | Path,
    reviewer: str,
    taxonomy: Taxonomy,
    console=None,
    ask: Callable[..., str] | None = None,
) -> list[Correction]:
    """Walk the queue with rich prompts. Each decision is appended to the audit log immediately."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    console = console or Console()
    ask = ask or (lambda *a, **k: Prompt.ask(*a, console=console, **k))
    done: list[Correction] = []
    valid = taxonomy.all_labels
    for n, item in enumerate(queue, 1):
        p = item.prediction
        rec = records[p.id]
        console.rule(f"[bold]{n}/{len(queue)}  {p.id}  ({p.source_model}, {p.domain})  priority={item.priority:.2f}")
        console.print(Panel(rec.prompt, title="prompt", expand=False))
        if rec.context:
            console.print(Panel(rec.context, title="context", expand=False))
        if rec.reference:
            console.print(Panel(rec.reference, title="reference", expand=False))
        console.print(Panel(rec.response, title="response", expand=False))
        t = Table("classifier", "labels", "evidence")
        t.add_row("heuristic", ", ".join(p.heuristic.labels), "; ".join(e for v in p.heuristic.evidence.values() for e in v)[:200])
        if p.llm is not None:
            llm_ev = p.llm.error or "; ".join(e for k, v in p.llm.evidence.items() if not k.startswith("_") for e in v)
            t.add_row("llm", ", ".join(p.llm.labels), llm_ev[:200])
        t.add_row("[bold]ensemble", "[bold]" + ", ".join(p.final_labels), f"confidence={p.confidence:.2f} agreement={p.agreement:.2f}")
        console.print(t)
        if item.reasons:
            console.print("[yellow]why queued:[/] " + "; ".join(item.reasons))

        choice = ask("[a]ccept  [c]orrect  [s]kip  [q]uit", choices=["a", "c", "s", "q"], default="a")
        if choice == "q":
            break
        if choice == "s":
            continue
        labels = None
        if choice == "c":
            while True:
                raw = ask(f"labels (comma separated; valid: {', '.join(valid)})")
                labels = [x.strip() for x in raw.split(",") if x.strip()]
                bad = [x for x in labels if x not in valid]
                if labels and not bad:
                    break
                console.print(f"[red]invalid labels: {bad or 'empty'}[/]")
        notes = ask("notes (optional)", default="")
        corr = make_correction(p, labels, reviewer, taxonomy, notes=notes, source="interactive")
        append_jsonl(corrections_path, [corr])
        done.append(corr)
        console.print(f"[green]{corr.action}: {corr.original_labels} -> {corr.corrected_labels}[/]")
    return done


def resolved_labels(
    predictions: list[Prediction],
    corrections: list[Correction],
    gold: dict[str, list[str]] | None = None,
) -> dict[str, tuple[list[str], str]]:
    """Best available label per item and its provenance: human > gold > model."""
    latest = latest_corrections(corrections)
    out = {}
    for p in predictions:
        if p.id in latest:
            out[p.id] = (latest[p.id].corrected_labels, "human")
        elif gold and p.id in gold:
            out[p.id] = (gold[p.id], "gold")
        else:
            out[p.id] = (p.final_labels, "model")
    return out


def correction_stats(corrections_path: str | Path | None) -> dict:
    corrs = load_corrections(corrections_path)
    latest = latest_corrections(corrs)
    changed = [c for c in latest.values() if c.action == "correct"]
    added: dict[str, int] = {}
    removed: dict[str, int] = {}
    for c in changed:
        for lab in set(c.corrected_labels) - set(c.original_labels) - {"none"}:
            added[lab] = added.get(lab, 0) + 1
        for lab in set(c.original_labels) - set(c.corrected_labels) - {"none"}:
            removed[lab] = removed.get(lab, 0) + 1
    return {
        "log_entries": len(corrs),
        "items_reviewed": len(latest),
        "accepted": len(latest) - len(changed),
        "corrected": len(changed),
        "labels_added": dict(sorted(added.items())),
        "labels_removed": dict(sorted(removed.items())),
        "reviewers": sorted({c.reviewer for c in corrs}),
    }
