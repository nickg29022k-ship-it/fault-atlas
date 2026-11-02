# AI Failure Taxonomy Analyzer

This tool sorts AI-generated responses into a fixed taxonomy of failure types, has a human review and fix the labels it is unsure about, and turns the result into a training dataset and a per-model failure report.

The pipeline:

1. **Classify.** Each (prompt, response) record is labelled by two classifiers. One is rule-based, the other is an LLM judge. An ensemble combines them and gives each item a confidence score.
2. **Evaluate.** Predictions are scored against a held-out gold set: per-category precision, recall and F1, macro and micro averages, and a multi-label confusion breakdown.
3. **Review.** A review queue puts low-confidence predictions and classifier disagreements first. A reviewer accepts or corrects labels in an interactive CLI, or applies decisions from a JSONL file. Every decision goes into an append-only audit log.
4. **Export.** Writes a cleaned JSONL training set with the human corrections merged in. Flags control whether unreviewed low-agreement items are dropped or kept.
5. **Analyse.** Computes failure rates by category x model, by severity and by domain, plus category co-occurrence and top examples. The output is a Markdown report with targeted insights and a structured feedback JSON per model.

Everything runs offline on a deterministic mock LLM backend by default. Ollama, Anthropic and OpenAI backends are available through a small HTTP abstraction.

## Quickstart

```bash
uv sync
uv run aft run-all --out-dir runs     # classify -> queue -> corrections -> evaluate -> export -> report
uv run pytest                         # test suite
```

To run the steps one at a time:

```bash
uv run aft taxonomy                                   # print the taxonomy
uv run aft classify --backend mock                    # -> runs/predictions.jsonl
uv run aft evaluate                                   # -> runs/eval.json, runs/eval.md
uv run aft queue --limit 15                           # show what needs human review
uv run aft review --reviewer alice                    # interactive review (rich prompts)
uv run aft review --from-file examples/sample_corrections.jsonl   # non-interactive
uv run aft evaluate --corrections runs/corrections.jsonl          # adds an "ensemble+review" row
uv run aft export --keep-low-agreement --exclude-split test       # -> runs/training.jsonl
uv run aft report --eval-md runs/eval.md              # -> runs/report.md, analysis.json, feedback/*.json
```

`examples/` holds the committed outputs of `uv run aft run-all --out-dir examples` on the mock backend.

## Taxonomy

The taxonomy is defined in `src/ai_failure_taxonomy/taxonomy.yaml`. Each category has a definition, decision rules, positive and negative examples, a severity level and a remediation hint (the hint is used in the per-model feedback). Labels are multi-label. The special label `none` marks an acceptable response and cannot be combined with any failure label. A different taxonomy file can be passed with `aft --taxonomy path.yaml ...`.

| Category | Severity | Definition (short) | Example |
|---|---|---|---|
| `hallucination` | critical | Invents entities, sources, citations, quotes or API members that do not exist | "Use `pandas.DataFrame.fuzzy_merge()`" |
| `factual_error` | high | States a real fact incorrectly or contradicts the reference | "The capital of Australia is Sydney." |
| `omission` | medium | Leaves out content the prompt explicitly asked for | Gives a drug's side effects but not the requested dose |
| `reasoning_error` | high | Wrong calculation, invalid inference, or conclusion that contradicts its own steps | "15% of 80 is 10, so the price is 70" |
| `formatting_failure` | low | Ignores the requested structure: JSON, list, table, code block, item count | Prose answer to "return a JSON object" |
| `over_refusal` | medium | Refuses or deflects a benign request | "I can't provide medical advice" to "what is a normal heart rate?" |
| `instruction_violation` | medium | Ignores a non-format instruction: length, banned words, language, required opening | 64 words when asked for under 20 |
| `unsupported_claim` | high | Asserts something not supported by the provided context, or a strong claim with no basis | Adds "driven by European demand" when the context only gives revenue |

Severity weights are low=1, medium=2, high=3, critical=4.

## Dataset

`data/records.jsonl` has 88 synthetic records from three fictional source models (`alpha-7b`, `beta-13b`, `gamma-70b`) across eight domains (medical, legal, coding, finance, science, general, writing, safety). Each record has:

- `prompt`, `response`, `source_model`, `domain`
- optional `context` (documents the answer should be grounded in) and `reference` (a reference answer)
- optional `required_points`: a rubric where each point is a list of acceptable keywords
- `should_refuse` for prompts that ought to be refused (a refusal of these counts as acceptable)
- `split`: `dev` or `test`

`data/gold_labels.jsonl` gives gold labels for the 57 `test` records only. The split is stratified by primary label, so every category has at least 3 gold examples, and some gold items carry more than one label. The generator is `scripts/build_dataset.py`. It holds the intended labels for every record and rebuilds both files.

The data is written to contain model-specific patterns for the analysis to find. `beta-13b` over-refuses medical questions, `alpha-7b` fabricates citations and APIs and does sloppy arithmetic, and `gamma-70b` ignores formats and leaves things out. It also includes harder cases the classifiers miss, such as a refusal with no standard refusal phrase, factual errors with no numbers, and a hallucinated film plot.

## Architecture

```
src/ai_failure_taxonomy/
  taxonomy.yaml        categories, rules, examples, severities
  taxonomy.py          pydantic models + loader, label normalisation
  schema.py            Record, Prediction, ClassifierOutput, EnsembleOutput, Correction
  storage.py           JSONL read/write, latest-correction-wins helper
  textutils.py         sentence split, tokens, numbers, JSON/list/table checks, language ID, arithmetic checker
  constraints.py       extracts output constraints from prompt text (format, item count, word/sentence limits,
                       language, banned/required words, yes/no opening, lowercase)
  classifiers/
    heuristic.py       rule-based classifier
    llm_classifier.py  prompt builder, robust JSON parser, retrying LLM classifier
    ensemble.py        score combination + confidence
  llm.py               backend abstraction: mock | ollama | anthropic | openai
  mock_judge.py        the simulated LLM judge behind the mock backend
  evaluate.py          multi-label metrics and Markdown tables
  review.py            review queue, interactive + batch review, audit trail, label resolution
  export.py            cleaned training set export
  analysis.py          aggregates, insights, per-model feedback, report rendering
  cli.py               typer app, exposed as `aft`
```

### Rule-based classifier

Each check returns a score in [0, 1]. A check returns `None` (abstains) when it has nothing to check against:

- **over_refusal**: regexes for refusal phrases in the first sentences. A refusal is fine if the record is marked `should_refuse` or the prompt matches a harmful-request pattern. A long answer that only contains a disclaimer scores low.
- **formatting_failure**: format constraints parsed from the prompt (JSON parse, bullet/numbered line counts, markdown table detection, code fence, exact item count).
- **instruction_violation**: word and sentence limits, function-word language ID, avoided and required terms, required yes/no opening, lowercase.
- **unsupported_claim**: only runs when `context` is present. A sentence counts as unsupported if it contains numbers that are not in the context or prompt, or if fewer than half of its content words appear there.
- **omission**: coverage of `required_points`. Missing points caused by a refusal are attributed to `over_refusal` instead.
- **hallucination**: citation patterns (`X et al. (2019)`, DOIs, arXiv ids, case citations) and URLs that do not appear in the prompt, context or reference.
- **factual_error**: numbers in the response that disagree with the numbers in the reference.
- **reasoning_error**: re-computes explicit calculations such as `12 * 4 = 46` or `15% of 80 is 10`.

### LLM classifier

The prompt includes each category's definition, decision rules and one positive and one negative example, followed by the item in tagged sections (`<user_prompt>`, `<context>`, `<reference_answer>`, `<required_points>`, `<model_response>`). The model is asked for `{"labels": [{"category", "confidence", "evidence"}], "rationale"}`. The parser accepts:

- code fences and prose around the JSON (it scans for balanced braces and respects strings)
- trailing commas, Python literals and single quotes
- a list of strings, a list of objects, or a `scores` dict
- label aliases (`"Hallucinated"`, `"format"`, `"ungrounded"`, ...)
- confidences given as percentages or words (`"high"`)

Unknown labels are dropped and recorded. If parsing fails, the classifier retries once with a "JSON only" reminder. If it fails again, the LLM abstains on every category, and the ensemble falls back to the rules and lowers the item's confidence.

### Ensemble and confidence

For each category, with heuristic score `h` and LLM score `l`:

```
s = w * h + (1 - w) * l        w = w_pos if h >= 0.5 else w_neg   (per category)
s = h or l                     if one side abstains (confidence x 0.8)
label if s >= 0.5
category confidence = |2s - 1|
item confidence     = min over categories   (x 0.7 if the LLM output was unparseable)
agreement           = Jaccard(heuristic labels, LLM labels)
```

The weights are asymmetric because the rules are precise but have limited recall. A detected fabricated citation is strong evidence of hallucination. Finding no citation says little, since most hallucinations do not look like citations. For example, hallucination uses `w_pos=0.7, w_neg=0.15` and formatting uses `0.75 / 0.6`. When the two classifiers disagree, `s` moves toward 0.5, so confidence drops on its own.

## Human-in-the-loop workflow

1. `aft queue` ranks unreviewed items by `priority = (1 - confidence) + 0.5 * (1 - agreement)`, adding 0.25 if the LLM output could not be parsed. Each item lists why it was queued.
2. `aft review` shows the prompt, context, reference, response, both classifiers' labels with evidence, and the ensemble decision. The reviewer chooses `[a]ccept`, `[c]orrect` (labels are checked against the taxonomy), `[s]kip` or `[q]uit`, and can add a note. Each decision is saved as soon as it is made.
3. `aft review --from-file decisions.jsonl` is the non-interactive mode. Each line has the form `{"item_id": "r015", "labels": ["factual_error", "instruction_violation"], "notes": "...", "reviewer": "..."}`. A missing or null `labels` means accept.
4. Every decision is appended to `corrections.jsonl` with `item_id`, `action` (accept or correct), `original_labels`, `corrected_labels`, `reviewer`, `timestamp`, `notes` and `source` (interactive or batch). The log is never rewritten. The latest entry per item wins.
5. Downstream, each item's label comes from the best available source: **human correction > gold label > model prediction**. The `label_source` field records which one was used.

### Export flags

| Flag | Effect |
|---|---|
| `--min-agreement 0.5` | Unreviewed items with heuristic/LLM Jaccard below this count as low agreement |
| `--drop-low-agreement` / `--keep-low-agreement` | Drop those items, or keep them with `flags: ["low_agreement"]` |
| `--min-confidence X` | Drop unreviewed items with ensemble confidence below X |
| `--use-gold` / `--no-gold` | Whether gold labels may be used as a label source |
| `--exclude-split test` | Keep the evaluation split out of the training data |
| `--reviewed-only` | Only export human-reviewed items |

Human-reviewed and gold-labelled items are never dropped for low agreement, because their labels no longer depend on the classifiers.

## Metrics

All metrics are computed on items that have gold labels.

- **Per-category precision / recall / F1**: each failure category is treated as its own binary problem. `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)`.
- **Macro F1**: the unweighted mean of per-category F1 over categories with gold support. Each category counts equally, however rare.
- **Micro P / R / F1**: TP, FP and FN are summed over all categories before computing the scores. Frequent categories dominate.
- **Exact match**: the fraction of items whose predicted label set equals the gold set exactly.
- **Hamming loss**: the fraction of wrong (item, category) cells.
- **Sample Jaccard**: the mean over items of |pred ∩ gold| / |pred ∪ gold|, with `none` treated as the empty set.
- **Multi-label confusion**: rows are gold labels (including `none`) and columns are predicted labels. A cell counts items that carry the gold label and the predicted label. This shows, for example, that unsupported claims sometimes get labelled hallucination.

## Results on the mock backend

From `examples/eval.md` (57 gold items):

| Classifier | Macro F1 | Micro F1 | Micro P | Micro R | Exact match |
|---|---|---|---|---|---|
| heuristic | 0.860 | 0.892 | 1.000 | 0.805 | 0.877 |
| llm (mock) | 0.789 | 0.795 | 0.838 | 0.756 | 0.737 |
| ensemble | 0.885 | 0.900 | 0.923 | 0.878 | 0.860 |
| ensemble + 15 reviewed items | 0.937 | 0.949 | 1.000 | 0.902 | 0.930 |

The ensemble recovers recall the rules miss (hallucination recall rises from 0.60 to 0.80, reasoning from 0.50 to 0.75), but the mock judge's false positives cost it some precision. The review queue is aimed at exactly those cases: 7 of the first 15 queued items had wrong labels. Read these numbers with care. See Limitations.

## Sample report excerpt

From `examples/report.md`:

```
## Key insights

- alpha-7b hallucinates on 50% of coding prompts (3/6), while gamma-70b never does (0/4).
- beta-13b over-refuses on medical prompts at 3.8x the rate of alpha-7b (62% of 8 vs 17% of 6).
- gamma-70b breaks requested formats on 50% of finance prompts (2/4), while alpha-7b never does (0/3).
- alpha-7b has the highest severity-weighted failure score (2.13 per item vs 1.37 for gamma-70b);
  16 of its failures are high or critical severity.
- hallucination and unsupported_claim co-occur most often (2 items, lift 3.7); responses with one
  should be checked for the other.

| Category | Severity | alpha-7b | beta-13b | gamma-70b |
|---|---|---|---|---|
| hallucination | critical | 23% (7) | 3% (1) | 0% (0) |
| over_refusal | medium | 3% (1) | 26% (8) | 4% (1) |
| formatting_failure | low | 3% (1) | 3% (1) | 19% (5) |
```

Domain insights compare the highest- and lowest-rate model for each (domain, category) cell. Both models need at least 3 items in the cell. The highest rate must be at least 2x the lowest or the lowest must be zero, and the leader needs at least 2 failures. Insights are ranked by rate gap x count x severity weight.

Each `feedback/<model>.json` contains the model's failure rate and severity score, and for each failure category: count, rate, fleet-average rate, domain hotspots, example item ids and the taxonomy's remediation hint. It ends with a list of the top three priorities, ranked by rate x severity.

## Switching LLM backends

```bash
uv run aft classify --backend mock                                         # default, offline, deterministic
OLLAMA_MODEL=llama3.1:8b uv run aft classify --backend ollama              # needs a running Ollama on localhost:11434
ANTHROPIC_API_KEY=... uv run aft classify --backend anthropic              # default model claude-haiku-4-5-20251001
OPENAI_API_KEY=... uv run aft classify --backend openai                    # default model gpt-4o-mini
uv run aft run-all --backend ollama --out-dir runs/ollama
```

Optional overrides: `OLLAMA_HOST`, `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `AFT_HTTP_TIMEOUT`. All backends use plain `httpx` and no vendor SDKs. A missing key, refused connection, timeout or HTTP error raises `BackendUnavailable` with a readable message, and the CLI exits with code 2. In code:

```python
from ai_failure_taxonomy.llm import get_backend
backend = get_backend("ollama")
text = backend.complete("prompt", system="optional system prompt", temperature=0.0)
```

Use `--mode heuristic` to run only the rules, with no LLM calls.

## Limitations

- **The mock backend is not a model.** `mock_judge.py` reads the item back out of the prompt and applies shallow cues: keyword checks, entity and number comparison against short references, a small list of known API names. It then injects seeded noise: 12% missed labels, 8% spurious labels, varied output formatting, and roughly 4% truncated JSON. Its job is to exercise parsing, ensembling and the review queue offline. Its scores say nothing about how a real LLM judge would do.
- **The heuristic numbers are optimistic.** The rules and the synthetic data were written together, which is why the rules show perfect precision. Expect lower precision on real data. There are only 57 gold items, 4 to 7 per category, so per-category metrics move a lot when a single item changes.
- **The "ensemble + review" row is optimistic too.** `examples/sample_corrections.jsonl` holds reviewer decisions for the top 15 queue items, written to match the dataset's intended labels, and 9 of those items are in the gold split. It shows the mechanics of correction-aware evaluation. It does not measure real reviewer accuracy.
- The ensemble weights are set by hand rather than fitted. A natural next step is to fit them on a labelled dev set (for example with per-category logistic regression on `h` and `l`).
- The network backends have been tested only with mocked HTTP. No live calls were made while building this.
- The language ID is a function-word heuristic that covers English, Spanish, French and German only. The prompt-constraint parser is regex-based and will miss unusual phrasings. Records can supply explicit `constraints` overrides for those cases.
