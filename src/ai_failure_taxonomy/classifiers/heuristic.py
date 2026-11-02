"""Rule-based failure classifier.

Each check returns a score in [0, 1] (probability-like strength of evidence)
or None when the check cannot be applied to the record (e.g. no context to
check grounding against). Labels are categories with score >= threshold.
"""

from __future__ import annotations

import re

from .. import textutils as tu
from ..constraints import Constraints, extract_constraints
from ..schema import ClassifierOutput, Record
from ..taxonomy import Taxonomy, load_taxonomy

REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I (?:can(?:'|no)t|cannot|won't|am unable to|am not able to)",
    r"\bI (?:can(?:'|no)t|cannot|won't|am unable to|am not able to) (?:help|assist|provide|give|answer|share|discuss|offer|comply)",
    r"\bI(?:'m| am) (?:not able|unable) to (?:help|assist|provide|give|answer|share|discuss|offer)",
    r"\bas an ai(?: language model)?,? I (?:can(?:'|no)t|cannot|don't|do not)",
    r"\bI(?:'m| am) not (?:able|allowed|permitted|qualified) to (?:provide|give|offer)",
    r"\b(?:it is|it's) not appropriate for me to",
    r"\bI must decline\b",
    r"\bplease consult (?:a|your) (?:doctor|physician|lawyer|attorney|professional|pharmacist|financial advisor|healthcare provider)\b",
]
_REFUSAL_RE = [re.compile(p, re.I) for p in REFUSAL_PATTERNS]

HARMFUL_PROMPT = re.compile(
    r"\b(bomb|explosive|nerve agent|bioweapon|untraceable (gun|poison)|synthesi[sz]e (meth|fentanyl)|"
    r"hack into|steal (a|someone's) (password|identity)|ransomware|kill (someone|my)|poison (my|someone))\b",
    re.I,
)

CITATION_RE = re.compile(
    r"([A-Z][a-z]+ et al\.,? \(?\d{4}\)?|\([A-Z][a-z]+(?: and [A-Z][a-z]+)?,? \d{4}\)|doi:\s*10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|"
    r"published in (?:the )?(?:Journal|Proceedings|Annals)[^.]*\d{4}|\b\d+ U\.S\. \d+|\b[A-Z][a-z]+ v\. [A-Z][\w&]+)"
)
URL_RE = re.compile(r"https?://[^\s)]+")
HEDGE_RE = re.compile(r"\b(the (context|document|passage|text) (does not|doesn't) (say|mention|specify|state))\b", re.I)


def detect_refusal(response: str) -> list[str]:
    head = " ".join(tu.sentences(response)[:3])
    return [m.group(0) for r in _REFUSAL_RE if (m := r.search(head))]


class HeuristicClassifier:
    name = "heuristic"

    def __init__(self, taxonomy: Taxonomy | None = None, threshold: float = 0.5):
        self.taxonomy = taxonomy or load_taxonomy()
        self.threshold = threshold

    # -- individual checks -------------------------------------------------
    def check_over_refusal(self, rec: Record, ev: list[str]) -> float:
        hits = detect_refusal(rec.response)
        if not hits:
            return 0.05
        harmful = rec.should_refuse or bool(HARMFUL_PROMPT.search(rec.prompt))
        if harmful:
            ev.append("refusal of a request that should be refused (acceptable)")
            return 0.02
        ev.append(f"refusal phrase: {hits[0]!r}")
        n = tu.word_count(rec.response)
        # A long answer that contains a disclaimer is usually a partial answer, not a refusal.
        if n > 90:
            ev.append(f"but response is long ({n} words), likely a caveat rather than a refusal")
            return 0.35
        return 0.92

    def check_formatting(self, rec: Record, c: Constraints, ev: list[str]) -> float:
        if not c.has_format:
            return 0.1
        resp = rec.response
        bullets, numbered = tu.list_items(resp)
        problems = []
        if c.format == "json" and not tu.is_valid_json(resp):
            problems.append("prompt requires JSON but response does not parse as JSON")
        elif c.format == "bullets" and bullets < 2:
            problems.append(f"prompt requires a bulleted list; found {bullets} bullet lines")
        elif c.format == "numbered" and numbered < 2:
            problems.append(f"prompt requires a numbered list; found {numbered} numbered lines")
        elif c.format == "table" and not tu.has_markdown_table(resp):
            problems.append("prompt requires a table; no markdown table found")
        elif c.format == "code_block" and "```" not in resp:
            problems.append("prompt requires a code block; none found")
        if c.num_items is not None and not problems:
            if c.format == "json":
                count = None
            else:
                count = max(bullets, numbered)
            if count is not None and count != c.num_items:
                problems.append(f"prompt asks for exactly {c.num_items} items; found {count}")
        ev.extend(problems)
        return 0.93 if problems else 0.06

    def check_instruction(self, rec: Record, c: Constraints, ev: list[str]) -> float:
        if not c.has_instruction:
            return 0.1
        resp = rec.response
        low = resp.lower()
        viol = []
        if c.max_words is not None:
            n = tu.word_count(resp)
            if n > c.max_words:
                viol.append(f"{n} words exceeds limit of {c.max_words}")
        if c.max_sentences is not None:
            n = len(tu.sentences(resp))
            if n > c.max_sentences:
                viol.append(f"{n} sentences exceeds limit of {c.max_sentences}")
        if c.language is not None:
            lang = tu.detect_language(resp)
            if lang not in (c.language, "unknown"):
                viol.append(f"response language looks like {lang}, expected {c.language}")
        for w in c.must_avoid:
            if re.search(rf"\b{re.escape(w.lower())}", low):
                viol.append(f"uses avoided term {w!r}")
        for w in c.must_include:
            if w.lower() not in low:
                viol.append(f"missing required term {w!r}")
        if c.must_start_with:
            first = re.sub(r"^[^A-Za-z]+", "", resp).split(maxsplit=1)
            first_word = first[0].lower().strip(".,!:;") if first else ""
            if first_word not in c.must_start_with:
                viol.append(f"response must start with one of {c.must_start_with}, starts with {first_word!r}")
        if c.lowercase and resp != resp.lower():
            viol.append("response must be all lowercase")
        ev.extend(viol)
        if not viol:
            return 0.07
        return 0.95 if len(viol) > 1 else 0.86

    def check_omission(self, rec: Record, refused: bool, ev: list[str]) -> float | None:
        if not rec.required_points:
            return None
        low = rec.response.lower()
        missing = [pt for pt in rec.required_points if not any(alt.lower() in low for alt in pt)]
        if not missing:
            return 0.05
        frac = len(missing) / len(rec.required_points)
        if refused:
            ev.append("required points missing, attributed to refusal")
            return 0.2
        ev.append(f"missing {len(missing)}/{len(rec.required_points)} required points: " + "; ".join(p[0] for p in missing))
        return min(0.97, 0.6 + 0.37 * frac)

    def check_unsupported(self, rec: Record, refused: bool, ev: list[str]) -> float | None:
        if not rec.context:
            return None
        if refused:
            return 0.05
        support_tokens = tu.content_tokens(rec.context) | tu.content_tokens(rec.prompt)
        support_nums = tu.numbers(rec.context) | tu.numbers(rec.prompt)
        unsupported = []
        for s in tu.sentences(rec.response):
            if HEDGE_RE.search(s):
                continue
            toks = tu.content_tokens(s)
            nums = tu.numbers(s) - support_nums
            if nums:
                unsupported.append((s, f"numbers {sorted(nums)} not in context"))
                continue
            if len(toks) >= 4:
                novel = toks - support_tokens
                cov = 1 - len(novel) / len(toks)
                if cov < 0.5 and len(novel) >= 3:
                    unsupported.append((s, f"only {cov:.0%} of content words appear in context"))
        for s, why in unsupported[:3]:
            ev.append(f"unsupported: {s[:90]!r} ({why})")
        if not unsupported:
            return 0.06
        return 0.78 if len(unsupported) == 1 else 0.9

    def check_hallucination(self, rec: Record, ev: list[str]) -> float:
        known = " ".join(filter(None, [rec.prompt, rec.context, rec.reference]))
        cites = [m.group(0) for m in CITATION_RE.finditer(rec.response) if m.group(0) not in known]
        urls = [u for u in URL_RE.findall(rec.response) if u not in known]
        if cites:
            ev.append(f"citation not traceable to prompt/context: {cites[0]!r}")
        if urls:
            ev.append(f"URL not present in prompt/context: {urls[0]!r}")
        if cites and urls:
            return 0.85
        if cites or urls:
            return 0.72
        return 0.2

    def check_factual(self, rec: Record, arith_errs: list[str], fabricated: bool, ev: list[str]) -> float | None:
        if not rec.reference:
            return None
        ref_nums = tu.numbers(rec.reference)
        if not ref_nums:
            return None
        resp_nums = tu.numbers(rec.response) - tu.numbers(rec.prompt)
        if not resp_nums:
            return 0.2
        if resp_nums & ref_nums:
            return 0.1
        if arith_errs:
            return 0.3  # the wrong number is explained by a reasoning slip
        if fabricated:
            return 0.35  # the numbers belong to a fabricated citation, not a misstated fact
        ev.append(f"numbers {sorted(resp_nums)[:4]} disagree with reference {sorted(ref_nums)[:4]}")
        return 0.7

    def check_reasoning(self, rec: Record, arith_errs: list[str], ev: list[str]) -> float | None:
        if arith_errs:
            ev.extend(f"arithmetic error: {e}" for e in arith_errs[:3])
            return 0.92
        if tu.ARITH.search(rec.response) or tu.PERCENT_OF.search(rec.response):
            return 0.1
        return None

    # -- public API -------------------------------------------------------
    def classify(self, rec: Record) -> ClassifierOutput:
        c = extract_constraints(rec.prompt, rec.constraints)
        ev: dict[str, list[str]] = {k: [] for k in self.taxonomy.ids}
        arith = tu.arithmetic_errors(rec.response)
        scores: dict[str, float | None] = {}
        scores["over_refusal"] = self.check_over_refusal(rec, ev["over_refusal"])
        refused = scores["over_refusal"] >= self.threshold or bool(detect_refusal(rec.response))
        scores["formatting_failure"] = self.check_formatting(rec, c, ev["formatting_failure"]) if not refused else 0.05
        scores["instruction_violation"] = self.check_instruction(rec, c, ev["instruction_violation"]) if not refused else 0.05
        scores["omission"] = self.check_omission(rec, refused, ev["omission"])
        scores["unsupported_claim"] = self.check_unsupported(rec, refused, ev["unsupported_claim"])
        scores["hallucination"] = self.check_hallucination(rec, ev["hallucination"])
        fabricated = scores["hallucination"] >= self.threshold
        scores["factual_error"] = self.check_factual(rec, arith, fabricated, ev["factual_error"])
        scores["reasoning_error"] = self.check_reasoning(rec, arith, ev["reasoning_error"])
        scores = {k: scores.get(k) for k in self.taxonomy.ids}
        labels = [k for k, v in scores.items() if v is not None and v >= self.threshold]
        return ClassifierOutput(
            name=self.name,
            scores={k: (round(v, 4) if v is not None else None) for k, v in scores.items()},
            labels=labels or [self.taxonomy.none_label],
            evidence={k: v for k, v in ev.items() if v},
        )
