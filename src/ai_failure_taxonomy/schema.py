"""Data models shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Record(BaseModel):
    """One (prompt, response) pair produced by a source model."""

    id: str
    prompt: str
    response: str
    source_model: str
    domain: str = "general"
    context: str | None = None  # documents the response should be grounded in
    reference: str | None = None  # reference answer, if one exists
    required_points: list[list[str]] = Field(default_factory=list)  # rubric: each point = keyword alternatives
    constraints: dict = Field(default_factory=dict)  # explicit overrides of constraints parsed from the prompt
    should_refuse: bool = False  # True if refusing is the correct behaviour
    split: Literal["dev", "test"] = "dev"


class GoldLabel(BaseModel):
    id: str
    labels: list[str]
    notes: str = ""


class ClassifierOutput(BaseModel):
    """Per-category scores in [0, 1]; None means the classifier abstained on that category."""

    name: str
    scores: dict[str, float | None]
    labels: list[str]
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    raw: str | None = None


class EnsembleOutput(BaseModel):
    scores: dict[str, float]
    labels: list[str]
    confidence: float  # 0..1, item-level
    category_confidence: dict[str, float]
    agreement: float  # Jaccard between heuristic and LLM label sets


class Prediction(BaseModel):
    id: str
    source_model: str
    domain: str
    heuristic: ClassifierOutput
    llm: ClassifierOutput | None = None
    ensemble: EnsembleOutput | None = None

    @property
    def final_labels(self) -> list[str]:
        if self.ensemble is not None:
            return self.ensemble.labels
        if self.llm is not None:
            return self.llm.labels
        return self.heuristic.labels

    @property
    def confidence(self) -> float:
        return self.ensemble.confidence if self.ensemble else 1.0

    @property
    def agreement(self) -> float:
        return self.ensemble.agreement if self.ensemble else 1.0


class Correction(BaseModel):
    """One audit-trail entry from a human reviewer."""

    item_id: str
    action: Literal["accept", "correct"]
    original_labels: list[str]
    corrected_labels: list[str]
    reviewer: str
    timestamp: str
    notes: str = ""
    source: Literal["interactive", "batch"] = "interactive"
