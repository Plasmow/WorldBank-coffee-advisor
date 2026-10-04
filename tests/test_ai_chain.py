"""The real chain (ai/analyze.py), with every model replaced by a stub.

The decision table is what keeps Noor safe, so it is tested here without
the translator, e5 or Ollama: fast, deterministic, and it runs on any laptop or CI.
The models themselves are exercised by tests/test_models.py, which skips
when they are not on disk.
"""

import pytest

CONTRACT = {"lang", "text_en", "label", "proba", "llm_label", "decision", "template_id", "reason"}


@pytest.fixture
def chain(monkeypatch):
    """ai.analyze with a scripted classifier, LLM and translator."""
    import ai.analyze as an

    script = {"clf": ("leaf_rust", 0.95), "llm": "leaf_rust", "question": "clarify_colour"}
    translated = []

    def fake_classify(text_en):
        return script["clf"](text_en) if callable(script["clf"]) else script["clf"]

    def fake_llm(text_en):
        if script["llm"] == "down":
            return {"label": None, "sure": False, "question": "clarify", "reason": "llm_error"}
        return {"label": script["llm"], "sure": True, "question": script["question"], "reason": "x"}

    def fake_translate(text):
        translated.append(text)
        return f"EN({text})"

    monkeypatch.setattr(an, "classify", fake_classify)
    monkeypatch.setattr(an, "llm_predict", fake_llm)
    monkeypatch.setattr(an.translate, "lug_to_en", fake_translate)
    script["translated"] = translated
    return an, script


def test_the_chain_imports_without_any_model():
    from ai.analyze import analyze  # noqa: F401


def test_agreement_above_threshold_answers_with_advice(chain):
    an, s = chain
    r = an.analyze("orange powder under the leaves")
    assert set(r) >= CONTRACT
    assert (r["decision"], r["template_id"], r["reason"]) == ("answer", "adv_leaf_rust", "agree")
    assert r["lang"] == "en"
    assert s["translated"] == []  # English needs no translation


def test_luganda_is_translated_before_classifying(chain):
    an, s = chain
    r = an.analyze("Ebikoola by'emmwanyi zange birina obuwunga")
    assert r["lang"] == "lg"
    assert r["text_en"].startswith("EN(")
    assert s["translated"] == ["Ebikoola by'emmwanyi zange birina obuwunga"]


def test_disagreement_asks_the_question_the_llm_chose(chain):
    an, s = chain
    s["llm"], s["question"] = "phoma", "clarify_rust_phoma"
    r = an.analyze("brown and orange marks on the leaves")
    assert (r["decision"], r["template_id"], r["reason"]) == ("clarify", "clarify_rust_phoma", "disagree")


def test_low_confidence_asks_one_question(chain):
    an, s = chain
    s["clf"], s["question"] = ("phoma", 0.55), "clarify_where"
    r = an.analyze("my coffee looks bad")
    assert (r["decision"], r["template_id"], r["reason"]) == ("clarify", "clarify_where", "low_confidence")


def test_still_unsure_after_the_question_escalates(chain):
    an, s = chain
    s["clf"] = ("phoma", 0.55)
    r = an.analyze("my coffee looks bad", clarify_answer="i do not know")
    assert (r["decision"], r["template_id"]) == ("escalate", "unsure")


def test_the_answer_is_read_with_the_first_message(chain):
    an, s = chain
    seen = []
    s["clf"] = lambda t: seen.append(t) or ("leaf_rust", 0.95)
    r = an.analyze("my coffee looks bad", clarify_answer="orange powder under the leaves")
    assert r["decision"] == "answer"
    assert "my coffee looks bad" in seen[-1] and "orange powder" in seen[-1]


def test_a_clear_answer_is_not_diluted_by_the_vague_first_message(chain):
    an, s = chain
    s["clf"] = lambda t: ("leaf_rust", 0.97) if t == "orange powder under the leaves" else ("leaf_rust", 0.7)
    r = an.analyze("my coffee looks bad", clarify_answer="orange powder under the leaves")
    assert (r["decision"], r["label"], r["proba"]) == ("answer", "leaf_rust", 0.97)


def test_a_vague_answer_does_not_rescue_a_vague_message(chain):
    an, s = chain
    s["clf"] = lambda t: ("other", 0.9) if t == "i do not know" else ("phoma", 0.6)
    r = an.analyze("my coffee looks bad", clarify_answer="i do not know")
    assert r["decision"] == "escalate"


def test_digits_are_never_sent_to_the_translator(chain):
    an, s = chain
    an.analyze("3")
    assert s["translated"] == []


def test_vague_other_gets_a_question_when_the_llm_is_unsure(chain, monkeypatch):
    # A badly translated Luganda message often looks like "other" to the
    # classifier; one question costs little, a needless agent call costs more.
    an, s = chain
    s["clf"] = ("other", 0.97)
    monkeypatch.setattr(an, "llm_predict", lambda t: {
        "label": "other", "sure": False, "question": "clarify_where", "reason": "vague"})
    r = an.analyze("Emmwanyi zange zirabika bubi")
    assert (r["decision"], r["template_id"]) == ("clarify", "clarify_where")


def test_confident_other_goes_straight_to_the_agent(chain):
    an, s = chain
    s["clf"], s["llm"] = ("other", 0.99), "other"
    r = an.analyze("insects are eating the berries")
    assert (r["decision"], r["template_id"], r["reason"]) == ("escalate", "unsure", "other")


def test_other_is_never_answered_whatever_the_llm_says(chain, monkeypatch):
    an, s = chain
    for proba in (0.5, 0.85, 0.99):
        for llm_label in ("other", "leaf_rust", "phoma", "healthy", None):
            for sure in (True, False):
                s["clf"] = ("other", proba)
                monkeypatch.setattr(an, "llm_predict", lambda t, l=llm_label, su=sure: {
                    "label": l, "sure": su, "question": "clarify", "reason": ""})
                for answer in (None, "more details"):
                    r = an.analyze("hello", clarify_answer=answer)
                    assert r["decision"] != "answer", (proba, llm_label, sure, answer)


def test_a_missing_llm_falls_back_to_the_classifier_alone(chain):
    an, s = chain
    s["llm"] = "down"
    r = an.analyze("orange powder under the leaves")
    assert r["llm_label"] is None
    assert (r["decision"], r["reason"], r["template_id"]) == ("answer", "llm_unavailable", "adv_leaf_rust")


def test_a_missing_llm_and_a_doubtful_classifier_still_ask(chain):
    an, s = chain
    s["llm"], s["clf"] = "down", ("leaf_rust", 0.6)
    r = an.analyze("orange marks")
    assert (r["decision"], r["template_id"]) == ("clarify", "clarify")


def test_a_broken_translation_falls_back_to_the_raw_text(chain, monkeypatch):
    an, _ = chain

    def boom(text):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(an.translate, "lug_to_en", boom)
    r = an.analyze("Ebikoola by'emmwanyi zange birina obuwunga")
    assert r["text_en"] == "Ebikoola by'emmwanyi zange birina obuwunga"


def test_every_template_the_chain_can_pick_exists():
    import json
    import pathlib

    from ai.analyze import ADVICE_TEMPLATES
    from ai.llm import QUESTIONS

    templates = json.loads((pathlib.Path(__file__).parent.parent / "data" / "templates.json")
                           .read_text(encoding="utf-8"))
    for key in [*ADVICE_TEMPLATES.values(), *QUESTIONS, "unsure"]:
        assert templates[key]["en"] and templates[key]["lg"], key


def test_llm_disabled_returns_no_label(monkeypatch):
    from ai import llm

    monkeypatch.setenv("USE_LLM", "0")
    assert llm.llm_predict("orange powder")["label"] is None


def test_llm_off_schema_answer_is_treated_as_unavailable(monkeypatch):
    from ai import llm

    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setattr(llm, "_ask", lambda t: '{"label": "coffee_wilt", "question": "nope"}')
    r = llm.llm_predict("something")
    assert r["label"] is None and r["question"] == "clarify"


def test_translate_exposes_both_names_the_team_uses():
    import ai.translate as tr

    assert tr.load is tr.preload


def test_a_missing_model_leaves_the_luganda_untouched(monkeypatch):
    """No download in CI, and no silence for the farmer either.

    lug_to_en must never raise: the glossary still feeds the classifier, and
    a clarifying question beats a dropped message.
    """
    import ai.translate as tr

    def no_model():
        raise FileNotFoundError("nothing on disk and nothing on the Hub")

    monkeypatch.setattr(tr, "_ensure_model", no_model)
    tr.get_translator.cache_clear()
    tr.get_tokenizer.cache_clear()
    tr._translate.cache_clear()
    monkeypatch.setattr(tr, "_warned", False)

    assert tr.lug_to_en("Ebikoola birina obuwunga") == "Ebikoola birina obuwunga"
    assert tr.is_loaded() is False
    # the glossary still carries the words the classifier keys on
    assert "powder" in tr.lug_to_en_with_terms("Ebikoola birina obuwunga")


def test_load_still_reports_the_failure_so_health_can_show_it(monkeypatch):
    import ai.translate as tr

    def no_model():
        raise FileNotFoundError("nothing on disk and nothing on the Hub")

    monkeypatch.setattr(tr, "_ensure_model", no_model)
    tr.get_translator.cache_clear()
    with pytest.raises(FileNotFoundError):
        tr.load()


def test_the_model_is_never_downloaded_just_to_translate_english(monkeypatch):
    import ai.translate as tr

    called = []
    monkeypatch.setattr(tr, "_ensure_model", lambda: called.append(1))
    tr._translate.cache_clear()
    tr.lug_to_en("")          # empty input short-circuits
    assert called == []


def test_en_to_lug_refuses_and_says_where_to_look():
    import ai.translate as tr

    with pytest.raises(NotImplementedError, match="templates.json"):
        tr.en_to_lug("It looks like the leaf has rust")

def test_the_translator_is_cached():
    import ai.translate as tr

    assert hasattr(tr.get_translator, "cache_info")


@pytest.mark.parametrize("text,lang", [
    ("orange powder under the leaves", "en"),
    ("my coffee plants look bad", "en"),
    ("Ebikoola by'emmwanyi zange birina obuwunga obwa langi ya kacungwa wansi waabyo", "lg"),
    ("Emmwanyi zigula ssente mmeka leero?", "lg"),
    ("Emmwanyi zange zirabika bubi", "lg"),
    ("Oli otya ssebo", "lg"),
])
def test_language_detection(text, lang):
    from ai.lang import detect_lang

    assert detect_lang(text) == lang


def test_language_detection_keeps_the_default_when_it_cannot_tell():
    from ai.lang import detect_lang

    assert detect_lang("zzzz", default="lg") == "lg"
    assert detect_lang("123", default="en") == "en"


def test_health_reports_whether_the_translator_is_in_memory(client):
    body = client.get("/health").json()
    assert body["translator_loaded"] is False  # nothing loaded a model in the tests


def test_asking_health_never_pulls_in_the_translation_model(client):
    import sys

    sys.modules.pop("ai.translate", None)
    client.get("/health")
    assert "ai.translate" not in sys.modules
