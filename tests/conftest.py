from pathlib import Path

import pytest

from ai_failure_taxonomy.schema import Record
from ai_failure_taxonomy.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "records.jsonl"
GOLD = ROOT / "data" / "gold_labels.jsonl"
SAMPLE_CORR = ROOT / "examples" / "sample_corrections.jsonl"


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


def make_record(**kw) -> Record:
    base = {"id": "x1", "prompt": "Say hi.", "response": "Hi!", "source_model": "m"}
    base.update(kw)
    return Record(**base)
