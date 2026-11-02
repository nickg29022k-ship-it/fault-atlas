"""Multi-label evaluation against gold labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .taxonomy import Taxonomy


@dataclass
class CategoryMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalResult:
    n_items: int
    per_category: dict[str, CategoryMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    exact_match: float
    hamming_loss: float
    sample_jaccard: float
    none_accuracy: float  # agreement on acceptable-vs-failure, ignoring which failure
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "macro": {"precision": self.macro_precision, "recall": self.macro_recall, "f1": self.macro_f1},
            "micro": {"precision": self.micro_precision, "recall": self.micro_recall, "f1": self.micro_f1},
            "exact_match": self.exact_match,
            "hamming_loss": self.hamming_loss,
            "sample_jaccard": self.sample_jaccard,
            "none_accuracy": self.none_accuracy,
            "per_category": {k: vars(v) for k, v in self.per_category.items()},
            "confusion": self.confusion,
            "errors": self.errors,
        }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def to_matrix(label_sets: list[list[str]], cats: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(label_sets), len(cats)), dtype=int)
    for r, labels in enumerate(label_sets):
        for lab in labels:
            if lab in idx:
                m[r, idx[lab]] = 1
    return m


def evaluate(
    gold: dict[str, list[str]],
    predicted: dict[str, list[str]],
    taxonomy: Taxonomy,
) -> EvalResult:
    """Compare predicted label sets with gold label sets on the items present in both.

    Per-category P/R/F1 are computed over the failure categories (one binary
    problem each). Macro averages over categories with gold support > 0;
    micro pools TP/FP/FN over all categories. The confusion breakdown
    counts, for each gold label (including `none`), how often each label was
    predicted on those items.
    """
    ids = sorted(set(gold) & set(predicted))
    if not ids:
        raise ValueError("no overlapping item ids between gold and predictions")
    cats = taxonomy.ids
    none = taxonomy.none_label
    Y = to_matrix([gold[i] for i in ids], cats)
    P = to_matrix([predicted[i] for i in ids], cats)

    tp = (Y & P).sum(0)
    fp = ((1 - Y) & P).sum(0)
    fn = (Y & (1 - P)).sum(0)
    tn = ((1 - Y) & (1 - P)).sum(0)
    per = {}
    for j, c in enumerate(cats):
        p, r, f = _prf(int(tp[j]), int(fp[j]), int(fn[j]))
        per[c] = CategoryMetrics(int(tp[j]), int(fp[j]), int(fn[j]), int(tn[j]), p, r, f, int(Y[:, j].sum()))

    supported = [c for c in cats if per[c].support > 0]
    macro_p = float(np.mean([per[c].precision for c in supported])) if supported else 0.0
    macro_r = float(np.mean([per[c].recall for c in supported])) if supported else 0.0
    macro_f = float(np.mean([per[c].f1 for c in supported])) if supported else 0.0
    mp, mr, mf = _prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))

    exact = float(np.mean([(Y[i] == P[i]).all() for i in range(len(ids))]))
    hamming = float((Y != P).mean())
    jac = []
    for i in range(len(ids)):
        u = (Y[i] | P[i]).sum()
        jac.append(1.0 if u == 0 else (Y[i] & P[i]).sum() / u)
    none_acc = float(np.mean([(Y[i].sum() == 0) == (P[i].sum() == 0) for i in range(len(ids))]))

    labels_all = cats + [none]
    confusion = {g: {p: 0 for p in labels_all} for g in labels_all}
    errors = []
    for i, item in enumerate(ids):
        g_set = [c for c in cats if Y[i, cats.index(c)]] or [none]
        p_set = [c for c in cats if P[i, cats.index(c)]] or [none]
        for g in g_set:
            for p in p_set:
                confusion[g][p] += 1
        if set(g_set) != set(p_set):
            errors.append({
                "id": item,
                "gold": g_set,
                "predicted": p_set,
                "missed": sorted(set(g_set) - set(p_set) - {none}),
                "spurious": sorted(set(p_set) - set(g_set) - {none}),
            })

    return EvalResult(
        n_items=len(ids),
        per_category=per,
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f, 4),
        micro_precision=mp,
        micro_recall=mr,
        micro_f1=mf,
        exact_match=round(exact, 4),
        hamming_loss=round(hamming, 4),
        sample_jaccard=round(float(np.mean(jac)), 4),
        none_accuracy=round(none_acc, 4),
        confusion=confusion,
        errors=errors,
    )


def format_eval_markdown(results: dict[str, EvalResult], taxonomy: Taxonomy) -> str:
    """Render one or more named evaluation results as Markdown tables."""
    names = list(results)
    lines = ["## Summary", "", "| Classifier | Macro F1 | Micro F1 | Micro P | Micro R | Exact match | Hamming loss | Jaccard |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        lines.append(f"| {n} | {r.macro_f1:.3f} | {r.micro_f1:.3f} | {r.micro_precision:.3f} | {r.micro_recall:.3f} | "
                     f"{r.exact_match:.3f} | {r.hamming_loss:.3f} | {r.sample_jaccard:.3f} |")
    lines += ["", "## Per-category F1", "", "| Category | Support | " + " | ".join(f"{n} P/R/F1" for n in names) + " |",
              "|---|---|" + "---|" * len(names)]
    first = results[names[0]]
    for c in taxonomy.ids:
        cells = []
        for n in names:
            m = results[n].per_category[c]
            cells.append(f"{m.precision:.2f} / {m.recall:.2f} / {m.f1:.2f}")
        lines.append(f"| {c} | {first.per_category[c].support} | " + " | ".join(cells) + " |")
    last = names[-1]
    labels = taxonomy.ids + [taxonomy.none_label]
    short = {c: c.split("_")[0][:6] for c in labels}
    lines += ["", f"## Multi-label confusion ({last})", "",
              "Rows are gold labels, columns are predicted labels; a cell counts items carrying both.", "",
              "| gold \\ pred | " + " | ".join(short[c] for c in labels) + " |", "|---|" + "---|" * len(labels)]
    conf = results[last].confusion
    for g in labels:
        lines.append(f"| {g} | " + " | ".join(str(conf[g][p]) for p in labels) + " |")
    return "\n".join(lines) + "\n"
