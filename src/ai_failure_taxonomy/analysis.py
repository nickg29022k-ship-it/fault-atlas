"""Aggregate failure analysis, targeted insights and per-model feedback."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .schema import Prediction, Record
from .taxonomy import SEVERITY_ORDER, Taxonomy

MIN_DOMAIN_N = 3  # minimum items per (model, domain) cell before comparing rates


@dataclass
class Analysis:
    models: list[str]
    domains: list[str]
    n_by_model: dict[str, int]
    n_by_model_domain: dict[str, dict[str, int]]
    failing_by_model_domain: dict[str, dict[str, int]]
    counts: dict[str, dict[str, int]]  # model -> category -> count
    rates: dict[str, dict[str, float]]
    failure_rate: dict[str, float]
    domain_counts: dict[str, dict[str, dict[str, int]]]  # model -> domain -> category -> count
    severity_counts: dict[str, dict[str, int]]  # model -> severity -> label count
    severity_score: dict[str, float]  # mean severity weight per item
    cooccurrence: dict[str, dict[str, int]]
    lift: dict[str, dict[str, float]]
    top_examples: dict[str, list[dict]]
    label_sources: dict[str, int]
    classifier_disagreement: dict[str, float]  # category -> rate of heuristic/LLM disagreement
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def analyze(
    records: list[Record],
    labels: dict[str, tuple[list[str], str]],
    taxonomy: Taxonomy,
    predictions: list[Prediction] | None = None,
    top_k: int = 3,
) -> Analysis:
    cats = taxonomy.ids
    recs = [r for r in records if r.id in labels]
    models = sorted({r.source_model for r in recs})
    domains = sorted({r.domain for r in recs})
    pred_by_id = {p.id: p for p in predictions or []}

    n_by_model = Counter(r.source_model for r in recs)
    n_md: dict[str, dict[str, int]] = {m: {d: 0 for d in domains} for m in models}
    counts = {m: {c: 0 for c in cats} for m in models}
    dcounts = {m: {d: {c: 0 for c in cats} for d in domains} for m in models}
    sev_counts = {m: {s: 0 for s in SEVERITY_ORDER} for m in models}
    sev_total = {m: 0.0 for m in models}
    failing = Counter()
    fail_md: dict[str, dict[str, int]] = {m: {d: 0 for d in domains} for m in models}
    co = np.zeros((len(cats), len(cats)), dtype=int)
    sources = Counter()
    by_cat_items: dict[str, list[tuple[float, Record, str]]] = defaultdict(list)

    for r in recs:
        labs, src = labels[r.id]
        sources[src] += 1
        fails = [c for c in labs if c in cats]
        n_md[r.source_model][r.domain] += 1
        if fails:
            failing[r.source_model] += 1
            fail_md[r.source_model][r.domain] += 1
        for c in fails:
            counts[r.source_model][c] += 1
            dcounts[r.source_model][r.domain][c] += 1
            sev = taxonomy.severity(c)
            sev_counts[r.source_model][sev] += 1
            sev_total[r.source_model] += taxonomy.severity_weights[sev]
            p = pred_by_id.get(r.id)
            score = p.ensemble.scores.get(c, 0.0) if p and p.ensemble else 0.0
            # Human/gold-labelled items first, then by model score.
            by_cat_items[c].append(((src != "model") + score, r, src))
        for a, b in combinations(sorted(fails, key=cats.index), 2):
            i, j = cats.index(a), cats.index(b)
            co[i, j] += 1
            co[j, i] += 1
        for a in fails:
            co[cats.index(a), cats.index(a)] += 1

    n_total = len(recs)
    lift = {}
    for i, a in enumerate(cats):
        lift[a] = {}
        for j, b in enumerate(cats):
            pa, pb, pab = co[i, i] / n_total, co[j, j] / n_total, co[i, j] / n_total
            lift[a][b] = round(float(pab / (pa * pb)), 2) if i != j and pa and pb else 0.0

    top = {}
    for c in cats:
        items = sorted(by_cat_items[c], key=lambda t: (-t[0], t[1].id))[:top_k]
        top[c] = [{
            "id": r.id, "source_model": r.source_model, "domain": r.domain, "label_source": src,
            "prompt": r.prompt[:140], "response": r.response[:200].replace("\n", " "),
        } for _, r, src in items]

    disagreement = {}
    if predictions:
        for c in cats:
            pairs = [(c in p.heuristic.labels, c in p.llm.labels) for p in predictions if p.llm is not None and p.llm.ok]
            disagreement[c] = round(sum(h != l for h, l in pairs) / len(pairs), 4) if pairs else 0.0

    an = Analysis(
        models=models,
        domains=domains,
        n_by_model=dict(n_by_model),
        n_by_model_domain=n_md,
        failing_by_model_domain=fail_md,
        counts=counts,
        rates={m: {c: round(counts[m][c] / n_by_model[m], 4) for c in cats} for m in models},
        failure_rate={m: round(failing[m] / n_by_model[m], 4) for m in models},
        domain_counts=dcounts,
        severity_counts=sev_counts,
        severity_score={m: round(sev_total[m] / n_by_model[m], 3) for m in models},
        cooccurrence={a: {b: int(co[i, j]) for j, b in enumerate(cats)} for i, a in enumerate(cats)},
        lift=lift,
        top_examples=top,
        label_sources=dict(sources),
        classifier_disagreement=disagreement,
    )
    an.insights = generate_insights(an, taxonomy)
    return an


def _verb(cat: str) -> str:
    return {
        "over_refusal": "over-refuses",
        "hallucination": "hallucinates",
        "factual_error": "makes factual errors",
        "omission": "omits required content",
        "reasoning_error": "makes reasoning errors",
        "formatting_failure": "breaks requested formats",
        "instruction_violation": "violates instructions",
        "unsupported_claim": "makes unsupported claims",
    }.get(cat, f"shows {cat}")


def generate_insights(an: Analysis, taxonomy: Taxonomy, max_domain: int = 5) -> list[str]:
    cats = taxonomy.ids
    scored: list[tuple[float, str]] = []

    # 1. Domain-specific gaps between models: "B over-refuses on medical prompts at 3x the rate of A".
    for d in an.domains:
        for c in cats:
            cells = [(m, an.domain_counts[m][d][c], an.n_by_model_domain[m][d]) for m in an.models
                     if an.n_by_model_domain[m][d] >= MIN_DOMAIN_N]
            if len(cells) < 2:
                continue
            rates = [(k / n, k, n, m) for m, k, n in cells]
            hi = max(rates)
            lo = min(rates)
            if hi[1] < 2 or hi[0] == lo[0]:
                continue
            if lo[0] > 0 and hi[0] / lo[0] < 2:
                continue
            ratio = f"{hi[0] / lo[0]:.1f}x the rate of" if lo[0] > 0 else "while never doing so for"
            text = (f"{hi[3]} {_verb(c)} on {d} prompts at {ratio} {lo[3]} "
                    f"({_pct(hi[0])} of {hi[2]} vs {_pct(lo[0])} of {lo[2]}).")
            if lo[0] == 0:
                text = (f"{hi[3]} {_verb(c)} on {_pct(hi[0])} of {d} prompts ({hi[1]}/{hi[2]}), "
                        f"while {lo[3]} never does ({lo[1]}/{lo[2]}).")
            scored.append(((hi[0] - lo[0]) * hi[1] * taxonomy.severity_weight(c), text))
    scored.sort(key=lambda t: -t[0])
    insights = [t for _, t in scored[:max_domain]]

    # 2. Overall category outliers.
    for c in cats:
        rates = {m: an.rates[m][c] for m in an.models}
        m_hi = max(rates, key=rates.get)
        others = [r for m, r in rates.items() if m != m_hi]
        mean_o = float(np.mean(others)) if others else 0.0
        if an.counts[m_hi][c] >= 3 and rates[m_hi] >= 2 * max(mean_o, 1e-9):
            rel = f"{rates[m_hi] / mean_o:.1f}x the other models' average" if mean_o > 0 else "and no other model does"
            insights.append(f"{m_hi} has the highest {c} rate: {_pct(rates[m_hi])} of its responses, {rel}.")

    # 3. Severity.
    worst = max(an.models, key=lambda m: an.severity_score[m])
    best = min(an.models, key=lambda m: an.severity_score[m])
    crit = {m: an.severity_counts[m]["critical"] + an.severity_counts[m]["high"] for m in an.models}
    insights.append(
        f"{worst} has the highest severity-weighted failure score ({an.severity_score[worst]:.2f} per item vs "
        f"{an.severity_score[best]:.2f} for {best}); {crit[worst]} of its failures are high or critical severity."
    )

    # 4. Co-occurrence.
    pairs = [(an.cooccurrence[a][b], a, b) for a, b in combinations(cats, 2) if an.cooccurrence[a][b] > 0]
    if pairs:
        k, a, b = max(pairs)
        insights.append(f"{a} and {b} co-occur most often ({k} items, lift {an.lift[a][b]:.1f}); "
                        f"responses with one should be checked for the other.")

    # 5. Classifier agreement.
    if an.classifier_disagreement:
        c = max(an.classifier_disagreement, key=an.classifier_disagreement.get)
        r = an.classifier_disagreement[c]
        if r > 0:
            insights.append(f"The rule-based and LLM classifiers disagree most on {c} ({_pct(r)} of items); "
                            f"labels in this category depend most on human review.")
    return insights


def model_feedback(an: Analysis, taxonomy: Taxonomy, model: str, examples_per_cat: int = 2) -> dict:
    cats = sorted(taxonomy.ids, key=lambda c: -an.rates[model][c])
    top = []
    for c in cats:
        if an.counts[model][c] == 0:
            continue
        hot = sorted(
            ((an.domain_counts[model][d][c] / n, d, an.domain_counts[model][d][c], n)
             for d, n in an.n_by_model_domain[model].items() if n and an.domain_counts[model][d][c]),
            reverse=True,
        )
        ex = [e["id"] for e in an.top_examples[c] if e["source_model"] == model][:examples_per_cat]
        top.append({
            "category": c,
            "count": an.counts[model][c],
            "rate": an.rates[model][c],
            "severity": taxonomy.severity(c),
            "fleet_rate": round(float(np.mean([an.rates[m][c] for m in an.models])), 4),
            "domain_hotspots": [{"domain": d, "rate": round(r, 4), "count": k, "n": n} for r, d, k, n in hot[:2]],
            "example_ids": ex,
            "recommendation": " ".join(taxonomy.get(c).remediation.split()),
        })
    priority = sorted(top, key=lambda t: -(t["rate"] * taxonomy.severity_weights[t["severity"]]))
    return {
        "model": model,
        "n_items": an.n_by_model[model],
        "failure_rate": an.failure_rate[model],
        "severity_weighted_score": an.severity_score[model],
        "severity_counts": an.severity_counts[model],
        "failures": top,
        "priorities": [t["category"] for t in priority[:3]],
        "insights": [i for i in an.insights if i.startswith(model)],
    }


def render_report(an: Analysis, taxonomy: Taxonomy, eval_md: str | None = None, review_stats: dict | None = None) -> str:
    cats = taxonomy.ids
    L = ["# Failure Analysis Report", ""]
    total = sum(an.n_by_model.values())
    src = ", ".join(f"{k}: {v}" for k, v in sorted(an.label_sources.items()))
    L += [f"Items analysed: {total} across {len(an.models)} models and {len(an.domains)} domains.",
          f"Label provenance (human correction > gold > model prediction): {src}.", ""]

    L += ["## Key insights", ""] + [f"- {i}" for i in an.insights] + [""]

    L += ["## Failure rate by model", "", "| Model | Items | Any failure | Severity score |", "|---|---|---|---|"]
    for m in an.models:
        L.append(f"| {m} | {an.n_by_model[m]} | {_pct(an.failure_rate[m])} | {an.severity_score[m]:.2f} |")
    L += ["", "Severity score = mean over items of the summed severity weights "
          f"({', '.join(f'{k}={v}' for k, v in taxonomy.severity_weights.items())}).", ""]

    L += ["## Failure rate by category and model", "",
          "| Category | Severity | " + " | ".join(an.models) + " |", "|---|---|" + "---|" * len(an.models)]
    for c in cats:
        cells = [f"{_pct(an.rates[m][c])} ({an.counts[m][c]})" for m in an.models]
        L.append(f"| {c} | {taxonomy.severity(c)} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Failures by severity", "", "| Model | " + " | ".join(SEVERITY_ORDER) + " |", "|---|" + "---|" * len(SEVERITY_ORDER)]
    for m in an.models:
        L.append(f"| {m} | " + " | ".join(str(an.severity_counts[m][s]) for s in SEVERITY_ORDER) + " |")
    L.append("")

    L += ["## Domain hotspots", "", "Items with at least one failure / items, and the most frequent category.", "",
          "| Domain | " + " | ".join(an.models) + " |", "|---|" + "---|" * len(an.models)]
    for d in an.domains:
        cells = []
        for m in an.models:
            n = an.n_by_model_domain[m][d]
            top_c = max(cats, key=lambda c: an.domain_counts[m][d][c])
            k = an.domain_counts[m][d][top_c]
            cells.append(f"{an.failing_by_model_domain[m][d]}/{n}" + (f", top: {top_c} ({k})" if k else "") if n else "-")
        L.append(f"| {d} | " + " | ".join(cells) + " |")
    L.append("")

    short = {c: c.split("_")[0][:6] for c in cats}
    L += ["## Category co-occurrence", "", "Diagonal = items with the category; off-diagonal = items with both.", "",
          "| | " + " | ".join(short[c] for c in cats) + " |", "|---|" + "---|" * len(cats)]
    for a in cats:
        L.append(f"| {a} | " + " | ".join(str(an.cooccurrence[a][b]) for b in cats) + " |")
    L.append("")

    L += ["## Top examples per category", ""]
    for c in cats:
        if not an.top_examples[c]:
            continue
        L.append(f"### {c}")
        for e in an.top_examples[c]:
            L.append(f"- `{e['id']}` {e['source_model']} / {e['domain']} ({e['label_source']}): "
                     f"*{e['prompt']}* -> \"{e['response'][:140]}\"")
        L.append("")

    if an.classifier_disagreement:
        L += ["## Classifier disagreement by category", "",
              "| Category | Heuristic vs LLM disagreement |", "|---|---|"]
        for c in cats:
            L.append(f"| {c} | {_pct(an.classifier_disagreement[c])} |")
        L.append("")

    if review_stats:
        L += ["## Human review", "",
              f"- Items reviewed: {review_stats['items_reviewed']} ({review_stats['accepted']} accepted, "
              f"{review_stats['corrected']} corrected) by {', '.join(review_stats['reviewers']) or 'n/a'}",
              f"- Labels added by reviewers: {review_stats['labels_added'] or 'none'}",
              f"- Labels removed by reviewers: {review_stats['labels_removed'] or 'none'}", ""]

    if eval_md:
        L += ["## Classifier evaluation (held-out gold set)", "", eval_md.replace("## ", "### ")]
    return "\n".join(L).rstrip() + "\n"
