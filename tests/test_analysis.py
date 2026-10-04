"""analyze(): one test per decision the provisional chain can reach.

Two things that look alike must not behave alike. A message about berries,
branches or insects is understood perfectly well -- it is simply outside the
three classes the model was trained on, so asking a question would waste the
farmer's one reply. A message we cannot read at all is worth exactly one
question before we give up and fetch a human.
"""

import pytest

from app.analysis import analyze


@pytest.fixture(autouse=True)
def keyword_chain(monkeypatch):
    monkeypatch.setenv("USE_REAL_AI", "0")


def test_healthy_answers_with_the_healthy_template():
    r = analyze("my coffee looks healthy, no problem")
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "healthy", "Healthy")


def test_rust_answers_with_the_rust_template():
    r = analyze("orange powder under the leaves")
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "leaf_rust", "Rust")


def test_phoma_answers_with_the_phoma_template():
    r = analyze("black lesions and dieback on my coffee")
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "phoma", "Phoma")


def test_not_understood_asks_one_question():
    r = analyze("zzzz qwerty")
    assert r["decision"] == "clarify"
    assert r["template_id"] == "clarify"


def test_not_understood_twice_escalates():
    r = analyze("zzzz qwerty", clarify_answer="still zzzz")
    assert r["decision"] == "escalate"
    assert r["template_id"] == "unsure"


@pytest.mark.parametrize(
    "message",
    [
        "my coffee berries are falling off",
        "the branches on my tree are drying",
        "small insects all over the stems",
    ],
)
def test_out_of_scope_escalates_without_asking_a_question(message):
    r = analyze(message)
    assert r["decision"] == "escalate", message
    assert r["label"] == "other"
    assert r["template_id"] == "unsure"


def test_the_clarification_is_read_together_with_the_first_message():
    # Neither half is a diagnosis on its own; together they are.
    assert analyze("something is wrong")["decision"] == "clarify"
    r = analyze("something is wrong", clarify_answer="orange powder on the leaves")
    assert (r["decision"], r["label"]) == ("answer", "leaf_rust")


def test_the_clarification_is_kept_in_the_english_text():
    r = analyze("something is wrong", clarify_answer="orange powder on the leaves")
    assert "something is wrong" in r["text_en"]
    assert "orange powder" in r["text_en"]


def test_the_unknown_template_is_never_proposed():
    for message in ("zzzz", "berries falling", "orange powder", "all good"):
        for answer in (None, "no idea"):
            assert analyze(message, clarify_answer=answer)["template_id"] != "Unknown"


def test_the_contract_keys_are_all_present():
    r = analyze("orange powder under the leaves")
    assert set(r) >= {"lang", "text_en", "label", "proba", "llm_label", "decision", "template_id"}
