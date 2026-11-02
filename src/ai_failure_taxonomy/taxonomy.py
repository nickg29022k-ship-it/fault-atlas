"""Loading and validating the failure taxonomy defined in YAML."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


class Category(BaseModel):
    id: str
    name: str
    severity: str
    definition: str
    decision_rules: list[str] = Field(min_length=1)
    positive_examples: list[str] = Field(min_length=1)
    negative_examples: list[str] = Field(min_length=1)
    remediation: str = ""

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str) -> str:
        if v not in SEVERITY_ORDER:
            raise ValueError(f"severity must be one of {SEVERITY_ORDER}, got {v!r}")
        return v


class Taxonomy(BaseModel):
    version: int
    none_label: str = "none"
    severity_weights: dict[str, int]
    categories: list[Category]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.categories]

    @property
    def all_labels(self) -> list[str]:
        return self.ids + [self.none_label]

    def get(self, cat_id: str) -> Category:
        for c in self.categories:
            if c.id == cat_id:
                return c
        raise KeyError(cat_id)

    def severity(self, cat_id: str) -> str:
        return self.get(cat_id).severity

    def severity_weight(self, cat_id: str) -> int:
        return self.severity_weights[self.severity(cat_id)]

    def normalize_labels(self, labels: list[str]) -> list[str]:
        """Validate labels, dedupe, order canonically and enforce `none` exclusivity."""
        unknown = [lab for lab in labels if lab not in self.all_labels]
        if unknown:
            raise ValueError(f"unknown labels: {unknown}; valid: {self.all_labels}")
        failures = [c for c in self.ids if c in labels]
        return failures if failures else [self.none_label]


def default_taxonomy_path() -> Path:
    return Path(str(resources.files("ai_failure_taxonomy").joinpath("taxonomy.yaml")))


@lru_cache(maxsize=8)
def _load_cached(path: str) -> Taxonomy:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    tax = Taxonomy.model_validate(raw)
    if len(set(tax.ids)) != len(tax.ids):
        raise ValueError("duplicate category ids in taxonomy")
    if tax.none_label in tax.ids:
        raise ValueError("none label must not also be a category")
    return tax


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    return _load_cached(str(Path(path) if path else default_taxonomy_path()))
