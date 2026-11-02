"""LLM-as-judge failure classifier with a structured JSON output contract."""

from __future__ import annotations

import json
import re

from ..llm import LLMBackend, get_backend
from ..mock_judge import TASK_MARKER
from ..schema import ClassifierOutput, Record
from ..taxonomy import Taxonomy, load_taxonomy

SYSTEM_PROMPT = f"""{TASK_MARKER}
You are a meticulous evaluator of AI assistant responses. You label each response
with zero or more failure categories from a fixed taxonomy. Apply the decision
rules exactly. Respond with a single JSON object and nothing else."""

OUTPUT_SPEC = """Return JSON with exactly this shape:
{
  "labels": [
    {"category": "<category id>", "confidence": <number 0-1>, "evidence": "<short quote or reason>"}
  ],
  "rationale": "<one or two sentences>"
}
Use only the category ids listed above. If the response is acceptable, return "labels": [].
Several categories may apply at once."""

ALIASES = {
    "hallucinated": "hallucination", "hallucinations": "hallucination", "fabrication": "hallucination",
    "fabricated": "hallucination", "made_up": "hallucination",
    "factual": "factual_error", "factual_errors": "factual_error", "factually_incorrect": "factual_error",
    "incorrect_fact": "factual_error", "fact_error": "factual_error",
    "omitted": "omission", "omissions": "omission", "incomplete": "omission", "missing_information": "omission",
    "reasoning": "reasoning_error", "logic_error": "reasoning_error", "logical_error": "reasoning_error",
    "math_error": "reasoning_error", "calculation_error": "reasoning_error",
    "formatting": "formatting_failure", "format": "formatting_failure", "format_error": "formatting_failure",
    "formatting_error": "formatting_failure", "format_failure": "formatting_failure",
    "refusal": "over_refusal", "overrefusal": "over_refusal", "over_refuse": "over_refusal",
    "unnecessary_refusal": "over_refusal",
    "instruction": "instruction_violation", "instruction_following": "instruction_violation",
    "instructions_violation": "instruction_violation", "constraint_violation": "instruction_violation",
    "unsupported": "unsupported_claim", "unsupported_claims": "unsupported_claim", "ungrounded": "unsupported_claim",
    "unsupported_statement": "unsupported_claim",
    "acceptable": "none", "no_failure": "none", "ok": "none", "pass": "none",
}

_WORD_CONF = {"very high": 0.95, "high": 0.85, "medium": 0.6, "moderate": 0.6, "low": 0.35, "very low": 0.15}


def build_prompt(rec: Record, taxonomy: Taxonomy) -> str:
    parts = ["# Taxonomy"]
    for c in taxonomy.categories:
        parts.append(f"## {c.id} (severity: {c.severity})\n{c.definition.strip()}")
        parts.append("Decision rules:\n" + "\n".join(f"- {r}" for r in c.decision_rules))
        parts.append(f"Example (is {c.id}): {c.positive_examples[0]}")
        parts.append(f"Example (is not {c.id}): {c.negative_examples[0]}")
    parts.append("\n# Item to evaluate")
    parts.append(f"<user_prompt>\n{rec.prompt}\n</user_prompt>")
    if rec.context:
        parts.append(f"<context>\n{rec.context}\n</context>")
    if rec.reference:
        parts.append(f"<reference_answer>\n{rec.reference}\n</reference_answer>")
    if rec.required_points:
        pts = "\n".join(f"- {' / '.join(p)}" for p in rec.required_points)
        parts.append(f"<required_points>\n{pts}\n</required_points>")
    if rec.should_refuse:
        parts.append("<note>This request is one the assistant should refuse.</note>")
    parts.append(f"<model_response>\n{rec.response}\n</model_response>")
    parts.append("\n# Output\n" + OUTPUT_SPEC)
    return "\n\n".join(parts)


def _extract_json_block(text: str) -> str | None:
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", text, re.S)
    if fence and fence.group(1).strip().startswith(("{", "[")):
        return fence.group(1).strip()
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced (e.g. truncated output)


def _loads_lenient(block: str):
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", block)  # trailing commas
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    if '"' not in fixed:
        fixed = fixed.replace("'", '"')
    return json.loads(fixed)


def normalize_category(name: str, valid: set[str]) -> str | None:
    key = re.sub(r"[\s\-/]+", "_", str(name).strip().lower()).strip("_")
    if key in valid or key == "none":
        return key
    return ALIASES.get(key)


def _conf(v, default: float = 0.75) -> float:
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower().rstrip("%")
        if s in _WORD_CONF:
            return _WORD_CONF[s]
        try:
            v = float(s)
        except ValueError:
            return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x > 1.0:
        x = x / 100.0
    return min(1.0, max(0.0, x))


def parse_llm_output(text: str, taxonomy: Taxonomy, absent_score: float = 0.15) -> ClassifierOutput:
    """Parse a judge response into per-category scores.

    Tolerates code fences, prose around the JSON, trailing commas, Python
    literals, label lists of strings or objects, a `scores` dict, label
    aliases and percentage / word confidences. Unknown labels are dropped.
    On failure returns ok=False with all scores None (abstain).
    """
    valid = set(taxonomy.ids)
    block = _extract_json_block(text or "")
    if block is None:
        return _failed(taxonomy, "no JSON object found in output", text)
    try:
        data = _loads_lenient(block)
    except json.JSONDecodeError as e:
        return _failed(taxonomy, f"invalid JSON: {e}", text)

    items: list[tuple[str, float, str]] = []
    if isinstance(data, list):
        data = {"labels": data}
    if not isinstance(data, dict):
        return _failed(taxonomy, "top-level JSON is not an object", text)
    default_conf = _conf(data.get("confidence"), 0.75)
    raw_labels = data.get("labels", data.get("categories", data.get("failures", [])))
    if isinstance(raw_labels, dict):  # {"hallucination": 0.9, ...}
        raw_labels = [{"category": k, "confidence": v} for k, v in raw_labels.items()]
    if isinstance(raw_labels, str):
        raw_labels = [s for s in re.split(r"[,;]", raw_labels) if s.strip()]
    for entry in raw_labels or []:
        if isinstance(entry, str):
            items.append((entry, default_conf, ""))
        elif isinstance(entry, dict):
            name = entry.get("category") or entry.get("label") or entry.get("id") or entry.get("name")
            if name is None:
                continue
            items.append((name, _conf(entry.get("confidence", entry.get("score")), default_conf), str(entry.get("evidence", ""))))
    if isinstance(data.get("scores"), dict):
        items.extend((k, _conf(v), "") for k, v in data["scores"].items())

    scores: dict[str, float | None] = {c: absent_score for c in taxonomy.ids}
    evidence: dict[str, list[str]] = {}
    dropped = []
    for name, conf, ev in items:
        cat = normalize_category(name, valid)
        if cat is None:
            dropped.append(str(name))
            continue
        if cat == "none":
            continue
        scores[cat] = max(scores[cat] if scores[cat] != absent_score else 0.0, conf)
        if ev:
            evidence.setdefault(cat, []).append(ev)
    rationale = data.get("rationale")
    if rationale:
        evidence.setdefault("_rationale", []).append(str(rationale))
    if dropped:
        evidence.setdefault("_dropped_labels", []).extend(dropped)
    labels = [c for c in taxonomy.ids if scores[c] is not None and scores[c] >= 0.5]
    return ClassifierOutput(
        name="llm", scores=scores, labels=labels or [taxonomy.none_label], evidence=evidence, raw=text
    )


def _failed(taxonomy: Taxonomy, err: str, raw: str | None) -> ClassifierOutput:
    return ClassifierOutput(
        name="llm", scores={c: None for c in taxonomy.ids}, labels=[taxonomy.none_label], ok=False, error=err, raw=raw
    )


class LLMClassifier:
    name = "llm"

    def __init__(self, backend: LLMBackend | str = "mock", taxonomy: Taxonomy | None = None, max_retries: int = 1):
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.taxonomy = taxonomy or load_taxonomy()
        self.max_retries = max_retries

    def classify(self, rec: Record) -> ClassifierOutput:
        prompt = build_prompt(rec, self.taxonomy)
        out = None
        for attempt in range(self.max_retries + 1):
            p = prompt if attempt == 0 else prompt + "\n\nYour previous output could not be parsed. Return ONLY the JSON object."
            text = self.backend.complete(p, system=SYSTEM_PROMPT, temperature=0.0)
            out = parse_llm_output(text, self.taxonomy)
            if out.ok:
                if attempt:
                    out.evidence.setdefault("_notes", []).append(f"parsed after {attempt} retry")
                return out
        return out  # type: ignore[return-value]
