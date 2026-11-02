"""Small text-processing helpers used by the heuristic classifier and the mock backend."""

from __future__ import annotations

import json
import re

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    this that these those it its it's they them their there here which who whom whose what when where why how
    not no yes do does did done can could should would will shall may might must have has had having
    i you he she we me my your our his her us also very more most such than too so just only about into over
    under after before between through during each any all some other both few many much same own out up down
    again further once per via while because based according context provided text document passage answer""".split()
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)(?:\s?%)?")


def sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line:
            continue
        parts.extend(s.strip() for s in SENT_SPLIT.split(line) if s.strip())
    return parts


def words(text: str) -> list[str]:
    return WORD.findall(text)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def stem(w: str) -> str:
    w = w.lower().strip("'-")
    for suf in ("ing", "edly", "ed", "ies", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_tokens(text: str) -> set[str]:
    return {stem(w) for w in words(text) if len(w) > 3 and w.lower() not in STOPWORDS}


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUMBER.finditer(text):
        n = m.group(1).replace(",", "")
        if n.endswith(".0"):
            n = n[:-2]
        out.add(n)
    return out


def strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", t, re.S)
    return m.group(1).strip() if m else t


def is_valid_json(text: str) -> bool:
    try:
        json.loads(strip_code_fence(text))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


BULLET_LINE = re.compile(r"^\s*[-*•]\s+\S")
NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+\S")


def list_items(text: str) -> tuple[int, int]:
    """Return (bullet_count, numbered_count)."""
    lines = text.splitlines()
    return (
        sum(bool(BULLET_LINE.match(l)) for l in lines),
        sum(bool(NUMBERED_LINE.match(l)) for l in lines),
    )


def has_markdown_table(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    pipe_lines = [l for l in lines if l.count("|") >= 2]
    has_sep = any(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", l) for l in lines)
    return len(pipe_lines) >= 3 and has_sep


# Very small language identifier based on function-word frequencies.
_LANG_WORDS = {
    "english": {"the", "and", "is", "are", "of", "to", "in", "it", "that", "you", "with", "for"},
    "spanish": {"el", "la", "los", "las", "es", "son", "de", "que", "y", "en", "un", "una", "por", "para", "con"},
    "french": {"le", "la", "les", "est", "sont", "de", "des", "et", "en", "un", "une", "pour", "avec", "que"},
    "german": {"der", "die", "das", "ist", "sind", "und", "zu", "ein", "eine", "mit", "für", "nicht", "den"},
}


def detect_language(text: str) -> str:
    toks = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]+", text)]
    if not toks:
        return "unknown"
    scores = {lang: sum(t in vocab for t in toks) / len(toks) for lang, vocab in _LANG_WORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.05 else "unknown"


ARITH = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)"
)
PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*of\s*\$?(\d+(?:\.\d+)?)\s*(?:is|=|equals|comes to)\s*\$?(\d+(?:\.\d+)?)", re.I
)


def arithmetic_errors(text: str) -> list[str]:
    """Find explicit calculations like '12 * 4 = 46' or '15% of 80 is 10' that are wrong."""
    errs = []
    for a, op, b, c in ARITH.findall(text):
        x, y, z = float(a), float(b), float(c.replace(",", ""))
        if op in "x×*":
            v = x * y
        elif op == "+":
            v = x + y
        elif op == "-":
            v = x - y
        else:
            if y == 0:
                continue
            v = x / y
        if abs(v - z) > max(0.01, 0.005 * abs(v)):
            errs.append(f"'{a} {op} {b} = {c}' (should be {v:g})")
    for p, base, res in PERCENT_OF.findall(text):
        v = float(p) / 100 * float(base)
        if abs(v - float(res)) > max(0.01, 0.005 * abs(v)):
            errs.append(f"'{p}% of {base} is {res}' (should be {v:g})")
    return errs
