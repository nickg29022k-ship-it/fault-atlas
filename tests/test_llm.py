import json

import pytest

from ai_failure_taxonomy.classifiers.llm_classifier import LLMClassifier, build_prompt, parse_llm_output
from ai_failure_taxonomy.llm import AnthropicBackend, BackendUnavailable, MockBackend, OllamaBackend, OpenAIBackend, get_backend

from .conftest import make_record

GOOD = {"labels": [{"category": "hallucination", "confidence": 0.9, "evidence": "fake paper"}], "rationale": "r"}


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here is my assessment:\n" + json.dumps(GOOD, indent=2) + "\nThanks!",
        '{"labels": [{"category": "hallucination", "confidence": 0.9,},], "rationale": "r",}',
        "{'labels': [{'category': 'Hallucinated', 'confidence': '90%'}]}",
        '{"labels": ["hallucination"], "confidence": 0.9}',
        '{"scores": {"hallucination": 0.9, "omission": 0.1}}',
        '[{"label": "fabrication", "score": "high"}]',
    ],
)
def test_parser_tolerates_common_output_shapes(taxonomy, text):
    out = parse_llm_output(text, taxonomy)
    assert out.ok, out.error
    assert out.labels == ["hallucination"]
    assert out.scores["hallucination"] >= 0.85


def test_parser_empty_labels_means_none(taxonomy):
    out = parse_llm_output('{"labels": [], "rationale": "fine"}', taxonomy)
    assert out.ok and out.labels == ["none"]
    assert all(v is not None and v < 0.5 for v in out.scores.values())


def test_parser_drops_unknown_labels(taxonomy):
    out = parse_llm_output('{"labels": ["toxicity", "omission"]}', taxonomy)
    assert out.labels == ["omission"]
    assert out.evidence["_dropped_labels"] == ["toxicity"]


@pytest.mark.parametrize("text", ["no json here", '{"labels": [{"category": "omission"', "", "42"])
def test_parser_failure_abstains(taxonomy, text):
    out = parse_llm_output(text, taxonomy)
    assert not out.ok and out.error
    assert all(v is None for v in out.scores.values())


def test_prompt_contains_taxonomy_and_item(taxonomy):
    rec = make_record(prompt="Q?", response="A.", context="CTX", reference="REF", required_points=[["a", "b"]])
    p = build_prompt(rec, taxonomy)
    for c in taxonomy.categories:
        assert c.id in p
    assert "<context>\nCTX\n</context>" in p and "<reference_answer>\nREF\n</reference_answer>" in p
    assert "- a / b" in p and '"labels"' in p


class FlakyBackend:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, system=None, temperature=0.0):
        self.calls += 1
        return "sorry, I cannot format that" if self.calls == 1 else '{"labels": ["omission"]}'


def test_llm_classifier_retries_on_parse_failure(taxonomy):
    be = FlakyBackend()
    out = LLMClassifier(be, taxonomy, max_retries=1).classify(make_record())
    assert be.calls == 2 and out.ok and out.labels == ["omission"]
    be2 = FlakyBackend()
    out2 = LLMClassifier(be2, taxonomy, max_retries=0).classify(make_record())
    assert be2.calls == 1 and not out2.ok


def test_mock_backend_is_deterministic_and_plausible(taxonomy):
    rec = make_record(prompt="What is a normal heart rate?", response="I'm sorry, I can't provide medical advice.")
    clf = LLMClassifier("mock", taxonomy)
    a, b = clf.classify(rec), clf.classify(rec)
    assert a.model_dump() == b.model_dump()
    assert MockBackend().complete("hello") == MockBackend().complete("hello")


def test_mock_backend_quality_on_dataset(taxonomy):
    """The simulated judge should be useful but imperfect."""
    import sys
    from pathlib import Path

    from ai_failure_taxonomy.storage import load_gold, load_records

    from .conftest import DATA, GOLD

    recs = {r.id: r for r in load_records(DATA)}
    gold = load_gold(GOLD)
    clf = LLMClassifier("mock", taxonomy)
    exact = sum(set(clf.classify(recs[i]).labels) == set(g.labels) for i, g in gold.items()) / len(gold)
    assert 0.5 < exact < 0.95


def test_get_backend():
    assert isinstance(get_backend("mock"), MockBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_network_backends_fail_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()
    with pytest.raises(BackendUnavailable, match="OPENAI_API_KEY"):
        OpenAIBackend()
    # Port 9 (discard) on localhost is not an Ollama server; connection should fail fast and clearly.
    with pytest.raises(BackendUnavailable, match="ollama"):
        OllamaBackend(host="http://127.0.0.1:9").complete("hi")


def test_backends_build_correct_requests(monkeypatch):
    import ai_failure_taxonomy.llm as llm

    seen = {}

    def fake_post(url, payload, headers, service):
        seen.update(url=url, payload=payload, headers=headers)
        if service == "anthropic":
            return {"content": [{"type": "text", "text": "ok-a"}]}
        if service == "openai":
            return {"choices": [{"message": {"content": "ok-o"}}]}
        return {"response": "ok-l"}

    monkeypatch.setattr(llm, "_post", fake_post)
    assert AnthropicBackend(api_key="k").complete("p", system="s") == "ok-a"
    assert seen["headers"]["x-api-key"] == "k" and seen["payload"]["system"] == "s"
    assert seen["payload"]["model"] == "claude-haiku-4-5-20251001"
    assert OpenAIBackend(api_key="k").complete("p", system="s") == "ok-o"
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "s"}
    monkeypatch.setenv("OLLAMA_MODEL", "tiny")
    assert OllamaBackend().complete("p") == "ok-l"
    assert seen["url"].endswith("/api/generate") and seen["payload"]["model"] == "tiny"
