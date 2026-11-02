"""Extract explicit output constraints from a prompt.

Constraints parsed from the prompt text can be overridden or extended with
the record's explicit `constraints` dict (useful when a prompt phrases a
constraint in a way the regexes do not catch).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMWORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"


def _n(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUMWORDS[tok.lower()]


@dataclass
class Constraints:
    format: str | None = None  # json | bullets | numbered | table | code_block
    num_items: int | None = None
    max_words: int | None = None
    max_sentences: int | None = None
    language: str | None = None
    must_include: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)
    must_start_with: list[str] = field(default_factory=list)  # any of these
    lowercase: bool = False

    @property
    def has_format(self) -> bool:
        return self.format is not None or self.num_items is not None

    @property
    def has_instruction(self) -> bool:
        return bool(
            self.max_words or self.max_sentences or self.language or self.must_include
            or self.must_avoid or self.must_start_with or self.lowercase
        )


def extract_constraints(prompt: str, overrides: dict | None = None) -> Constraints:
    p = prompt
    pl = prompt.lower()
    c = Constraints()

    if re.search(r"\b(valid json|json object|json array|as json|in json|json format|return json|respond with json|output json)\b", pl):
        c.format = "json"
    elif re.search(r"\b(markdown table|as a table|in a table|table with)\b", pl):
        c.format = "table"
    elif re.search(r"\bnumbered (list|steps)\b", pl):
        c.format = "numbered"
    elif re.search(r"\b(bullet(ed)? (points?|list)|as bullets)\b", pl):
        c.format = "bullets"
    elif re.search(r"\b(in a code block|inside a code block|fenced code block)\b", pl):
        c.format = "code_block"

    m = re.search(rf"\b(?:exactly|list|give|name|provide|top)\s+(?:me\s+)?{_NUM}\s+(?:[\w-]+\s+){{0,2}}?(bullet points?|bullets|items|steps|reasons|tips|examples|points|ways|benefits|risks|ideas|options|types|kinds|methods|elements|signs|things|facts|colors|names)\b", pl)
    if m:
        c.num_items = _n(m.group(1))

    m = re.search(rf"\b(?:under|fewer than|less than|at most|no more than|maximum of|max(?:imum)?|within)\s+{_NUM}\s+words\b", pl) or re.search(
        rf"\bin\s+{_NUM}\s+words or (?:fewer|less)\b", pl
    )
    if m:
        n = _n(m.group(1))
        strict = re.search(r"\b(under|fewer than|less than)\s", m.group(0)) and not re.search(r"\bor (fewer|less)\b", m.group(0))
        c.max_words = n - 1 if strict else n

    m = re.search(rf"\b(?:in|using|at most|no more than)\s+(?:exactly\s+)?(?:a single|{_NUM})\s+sentences?\b", pl)
    if m:
        c.max_sentences = 1 if "single" in m.group(0) else _n(m.group(1))
    elif re.search(r"\bin one sentence\b|\bone-sentence\b", pl):
        c.max_sentences = 1

    m = re.search(r"\b(?:in|respond in|answer in|reply in|write in)\s+(spanish|french|german|english)\b", pl)
    if m:
        c.language = m.group(1)

    for m in re.finditer(r"(?:do not|don't|never|without)\s+(?:mention(?:ing)?|use|using|say|saying|include|including)\s+(?:the word\s+|the term\s+)?[\"']?([A-Za-z][\w\- ]{1,30}?)[\"']?(?=[.,;!?]|$| or | and )", p, re.I):
        c.must_avoid.append(m.group(1).strip())
    for m in re.finditer(r"(?:avoid|no)\s+(?:the word|the term|jargon like)\s+[\"']?([A-Za-z][\w\-]+)[\"']?", p, re.I):
        c.must_avoid.append(m.group(1).strip())
    for m in re.finditer(r"(?:must|should|be sure to|make sure to)\s+(?:mention|include|use)\s+(?:the word\s+|the term\s+)?[\"']([^\"']+)[\"']", p, re.I):
        c.must_include.append(m.group(1).strip())

    if re.search(r"\b(start|begin) (your answer |your response )?with (a )?(yes or no|\"yes\" or \"no\")", pl) or re.search(r"answer yes or no first", pl):
        c.must_start_with = ["yes", "no"]
    if re.search(r"\b(all lowercase|in lowercase|lower-case only)\b", pl):
        c.lowercase = True

    for k, v in (overrides or {}).items():
        if not hasattr(c, k):
            raise ValueError(f"unknown constraint override: {k}")
        setattr(c, k, v)
    return c
