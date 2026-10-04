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
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "healthy", "adv_healthy")


def test_rust_answers_with_the_rust_template():
    r = analyze("orange powder under the leaves")
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "leaf_rust", "adv_leaf_rust")


def test_phoma_answers_with_the_phoma_template():
    r = analyze("black lesions and dieback on my coffee")
    assert (r["decision"], r["label"], r["template_id"]) == ("answer", "phoma", "adv_phoma")


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


@pytest.mark.parametrize(
    "message",
    [
        # the four quick-reply chips on the demo page, verbatim
        "The leaves on the upper trees have orange powder underneath",
        "Dark brown spots on the leaves and they dry and fall after the cold nights",
        "Something is wrong with my coffee trees",
        "The berries have small holes",
    ],
)
def test_every_demo_chip_reaches_a_sensible_decision(message):
    r = analyze(message)
    assert r["decision"] in ("answer", "clarify", "escalate")
    assert r["template_id"] != "Unknown"


def test_brown_spots_read_as_phoma():
    # The demo page offers this exact wording; it must not fall through to a
    # clarifying question the scenario never answers.
    r = analyze("Dark brown spots on the leaves and they dry and fall after the cold nights")
    assert (r["decision"], r["label"]) == ("answer", "phoma")


def test_nllb_calling_coffee_beans_does_not_send_everyone_to_an_agent():
    # Real NLLB output: "Ebikoola by'emmwanyi zange birina obutuli obwa kyenvu"
    # ("the leaves of my coffee have yellow spots") comes back as "My coffee
    # beans are yellowish". It loses the leaves and invents beans. With "bean"
    # treated as out of scope, every translated message would reach a human
    # and the classifier would never be asked anything.
    r = analyze("My coffee beans are yellowish")
    assert r["decision"] == "clarify"


def test_fruit_problems_still_reach_a_human():
    for message in ("my coffee berries are falling", "the cherries are rotting"):
        assert analyze(message)["decision"] == "escalate", message
