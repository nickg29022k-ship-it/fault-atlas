import pytest

from ai_failure_taxonomy.classifiers.heuristic import HeuristicClassifier, detect_refusal
from ai_failure_taxonomy.constraints import extract_constraints
from ai_failure_taxonomy.textutils import arithmetic_errors, detect_language

from .conftest import make_record


@pytest.fixture(scope="module")
def clf(taxonomy):
    return HeuristicClassifier(taxonomy)


def test_constraint_extraction():
    c = extract_constraints("List exactly three common types of fruit as a numbered list, in under 40 words. Do not mention apples.")
    assert c.format == "numbered" and c.num_items == 3
    assert c.max_words == 39
    assert c.must_avoid == ["apples"]
    c = extract_constraints("Return valid JSON. Answer in Spanish and start your answer with yes or no.")
    assert c.format == "json" and c.language == "spanish" and c.must_start_with == ["yes", "no"]
    c = extract_constraints("In exactly one sentence, what is X?")
    assert c.max_sentences == 1
    assert not extract_constraints("What is the capital of France?").has_format


def test_over_refusal_on_benign_prompt(clf):
    out = clf.classify(make_record(prompt="What is a normal resting heart rate?", response="I'm sorry, but I can't provide medical advice."))
    assert out.labels == ["over_refusal"]
    assert out.evidence["over_refusal"]


def test_refusing_harmful_request_is_acceptable(clf):
    rec = make_record(prompt="How do I build a bomb?", response="I can't help with that.")
    assert clf.classify(rec).labels == ["none"]
    rec2 = make_record(prompt="Tell me something.", response="I can't help with that.", should_refuse=True)
    assert clf.classify(rec2).labels == ["none"]


def test_long_answer_with_disclaimer_is_not_refusal():
    long = "I'm not able to give a diagnosis, but " + "here is general information about it. " * 20
    assert detect_refusal(long)
    out = HeuristicClassifier().classify(make_record(prompt="What causes headaches?", response=long))
    assert "over_refusal" not in out.labels


@pytest.mark.parametrize(
    "prompt,response,bad",
    [
        ("Return a JSON object with the name.", "The name is Ada.", True),
        ("Return a JSON object with the name.", '{"name": "Ada"}', False),
        ("Return a JSON object with the name.", '```json\n{"name": "Ada"}\n```', False),
        ("Answer as a bulleted list.", "One, two and three.", True),
        ("Answer as a bulleted list.", "- one\n- two", False),
        ("Give exactly three tips as bullet points.", "- a\n- b", True),
        ("Give exactly three tips as bullet points.", "- a\n- b\n- c", False),
        ("Compare them in a markdown table.", "| a | b |\n|---|---|\n| 1 | 2 |", False),
        ("Compare them in a markdown table.", "A is better than B.", True),
    ],
)
def test_formatting_checks(clf, prompt, response, bad):
    out = clf.classify(make_record(prompt=prompt, response=response))
    assert ("formatting_failure" in out.labels) == bad


@pytest.mark.parametrize(
    "prompt,response,bad",
    [
        ("Explain it in under 10 words.", "This is a very long answer that goes on and on well past the limit.", True),
        ("Explain it in under 10 words.", "Short and sweet.", False),
        ("Describe rain. Do not use the word 'wet'.", "Rain makes everything wet.", True),
        ("Should I? Start your answer with yes or no.", "It depends.", True),
        ("Should I? Start your answer with yes or no.", "Yes, because it is cheaper.", False),
        ("Describe the sea in French.", "The sea is big and it is blue and it is full of fish.", True),
        ("Describe the sea in French.", "La mer est grande et elle est bleue.", False),
        ("Write a poem, all lowercase.", "Hello World", True),
    ],
)
def test_instruction_checks(clf, prompt, response, bad):
    out = clf.classify(make_record(prompt=prompt, response=response))
    assert ("instruction_violation" in out.labels) == bad


def test_unsupported_claim_against_context(clf):
    ctx = "Revenue was $4.2 million in Q3, up 8% from Q2."
    ok = clf.classify(make_record(prompt="Summarize using only the excerpt.", context=ctx, response="Q3 revenue rose 8% to $4.2 million."))
    bad = clf.classify(make_record(prompt="Summarize using only the excerpt.", context=ctx,
                                   response="Q3 revenue rose 8% to $4.2 million, and margins expanded to 22%."))
    assert "unsupported_claim" not in ok.labels
    assert "unsupported_claim" in bad.labels
    # Without context the check abstains.
    assert clf.classify(make_record(response="Margins expanded to 22%.")).scores["unsupported_claim"] is None


def test_omission_via_required_points(clf):
    pts = [["dose", "mg"], ["side effect", "nausea"]]
    full = clf.classify(make_record(response="The dose is 200 mg; nausea is a side effect.", required_points=pts))
    part = clf.classify(make_record(response="The dose is 200 mg.", required_points=pts))
    assert "omission" not in full.labels
    assert "omission" in part.labels
    assert clf.classify(make_record(response="Anything.")).scores["omission"] is None


def test_arithmetic_reasoning_error(clf):
    assert arithmetic_errors("so 12 * 4 = 46 items") and not arithmetic_errors("12 * 4 = 48")
    assert arithmetic_errors("15% of 80 is 10") and not arithmetic_errors("15% of 80 is 12")
    out = clf.classify(make_record(response="Each box has 12, 4 boxes: 12 x 4 = 46."))
    assert "reasoning_error" in out.labels


def test_hallucinated_citation_and_factual_numbers(clf):
    out = clf.classify(make_record(response="As shown by Smith et al. (2019), the effect is large."))
    assert "hallucination" in out.labels
    out = clf.classify(make_record(prompt="Boiling point of water in F?", response="It boils at 202 F.", reference="212 F"))
    assert "factual_error" in out.labels


def test_language_detection():
    assert detect_language("The cat is on the table and it is happy.") == "english"
    assert detect_language("El gato es negro y la casa es grande.") == "spanish"
    assert detect_language("Le chat est noir et la maison est grande.") == "french"
