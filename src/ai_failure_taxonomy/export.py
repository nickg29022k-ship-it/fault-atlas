"""Export a cleaned training dataset that merges human corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .review import resolved_labels
from .schema import Correction, Prediction, Record
from .storage import latest_corrections, write_jsonl
from .taxonomy import SEVERITY_ORDER, Taxonomy


@dataclass
class ExportOptions:
    min_agreement: float = 0.5  # heuristic/LLM Jaccard below this = low agreement
    min_confidence: float = 0.0  # ensemble confidence below this = low confidence
    drop_low_agreement: bool = True  # drop (True) or keep-with-flag (False) unreviewed low-agreement items
    use_gold: bool = True  # use gold labels when no human correction exists
    exclude_splits: list[str] = field(default_factory=list)  # e.g. ["test"] to keep the eval set out of training
    reviewed_only: bool = False


@dataclass
class ExportStats:
    total: int = 0
    written: int = 0
    dropped_low_agreement: int = 0
    dropped_low_confidence: int = 0
    dropped_split: int = 0
    dropped_unreviewed: int = 0
    kept_flagged: int = 0
    label_sources: dict[str, int] = field(default_factory=dict)


def max_severity(labels: list[str], taxonomy: Taxonomy) -> str | None:
    sev = [taxonomy.severity(l) for l in labels if l in taxonomy.ids]
    return max(sev, key=SEVERITY_ORDER.index) if sev else None


def build_training_rows(
    records: list[Record],
    predictions: list[Prediction],
    corrections: list[Correction],
    taxonomy: Taxonomy,
    gold: dict[str, list[str]] | None = None,
    opts: ExportOptions | None = None,
) -> tuple[list[dict], ExportStats]:
    opts = opts or ExportOptions()
    rec_by_id = {r.id: r for r in records}
    latest = latest_corrections(corrections)
    resolved = resolved_labels(predictions, corrections, gold if opts.use_gold else None)
    stats = ExportStats(total=len(predictions))
    rows = []
    for p in predictions:
        rec = rec_by_id[p.id]
        labels, source = resolved[p.id]
        if rec.split in opts.exclude_splits:
            stats.dropped_split += 1
            continue
        if opts.reviewed_only and source != "human":
            stats.dropped_unreviewed += 1
            continue
        flags = []
        if source == "model":  # human and gold labels override classifier uncertainty
            if p.agreement < opts.min_agreement:
                if opts.drop_low_agreement:
                    stats.dropped_low_agreement += 1
                    continue
                flags.append("low_agreement")
            if p.confidence < opts.min_confidence:
                stats.dropped_low_confidence += 1
                continue
        if flags:
            stats.kept_flagged += 1
        stats.label_sources[source] = stats.label_sources.get(source, 0) + 1
        corr = latest.get(p.id)
        rows.append({
            "id": rec.id,
            "prompt": rec.prompt,
            "response": rec.response,
            "context": rec.context,
            "reference": rec.reference,
            "source_model": rec.source_model,
            "domain": rec.domain,
            "labels": labels,
            "is_failure": labels != [taxonomy.none_label],
            "max_severity": max_severity(labels, taxonomy),
            "label_source": source,
            "model_labels": p.final_labels,
            "confidence": p.confidence,
            "agreement": p.agreement,
            "flags": flags,
            "reviewer": corr.reviewer if corr else None,
            "review_notes": corr.notes if corr else None,
        })
    stats.written = len(rows)
    return rows, stats


def export_training(path: str | Path, *args, **kwargs) -> ExportStats:
    rows, stats = build_training_rows(*args, **kwargs)
    write_jsonl(path, rows)
    return stats
