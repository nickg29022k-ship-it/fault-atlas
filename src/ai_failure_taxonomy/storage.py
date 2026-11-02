"""JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from .schema import Correction, GoldLabel, Prediction, Record

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict | BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            d = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict | BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            d = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def _load(path: str | Path, model: type[T]) -> list[T]:
    return [model.model_validate(r) for r in read_jsonl(path)]


def load_records(path: str | Path) -> list[Record]:
    return _load(path, Record)


def load_gold(path: str | Path) -> dict[str, GoldLabel]:
    return {g.id: g for g in _load(path, GoldLabel)}


def load_predictions(path: str | Path) -> list[Prediction]:
    return _load(path, Prediction)


def load_corrections(path: str | Path | None) -> list[Correction]:
    if path is None or not Path(path).exists():
        return []
    return _load(path, Correction)


def latest_corrections(corrections: list[Correction]) -> dict[str, Correction]:
    """The audit log is append-only; the latest entry per item wins."""
    out: dict[str, Correction] = {}
    for c in sorted(corrections, key=lambda c: c.timestamp):
        out[c.item_id] = c
    return out
