"""Noor writes Luganda and gets Luganda; any question about money gets prices.json."""

import json
import pathlib
import string

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATES = json.loads((ROOT / "data" / "templates.json").read_text(encoding="utf-8"))
LG_PHONE = "+256799000301"


def send(client, phone, text):
    r = client.post("/api/demo/send", json={"phone": phone, "text": text})
    assert r.status_code == 200, r.text
    return r.json()


def _boom(*a, **k):
    raise AssertionError("a price question must not reach the AI chain")


# ---------- templates.json ----------

@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_has_english_and_luganda(key):
    v = TEMPLATES[key]
    assert v["en"].strip() and v["lg"].strip()
    assert "lg_verified" in v


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_fits_one_sms(key):
    for lang in ("en", "lg"):
        assert len(TEMPLATES[key][lang]) <= 160, (key, lang)


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_both_languages_use_the_same_placeholders(key):
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}
    assert fields(TEMPLATES[key]["en"]) == fields(TEMPLATES[key]["lg"]), key


def test_the_price_message_fits_one_sms_in_both_languages():
    from app import content

    for lang in ("en", "lg"):
        text = content.prices_text(lang)
        assert len(text) <= 160, text
        assert "UGX" in text and "15,000-16,000" in text


# ---------- language ----------

def test_luganda_message_gets_luganda_replies(client):
    r = send(client, LG_PHONE, "Ebikoola by'emmwanyi zange birina obubonero obutategeerekeka")
    assert TEMPLATES["welcome"]["lg"] in r["replies"]
    # the keyword chain cannot read Luganda: one question, asked in Luganda
    assert TEMPLATES["clarify"]["lg"] in r["replies"]


def test_english_message_gets_english_replies(client):
    r = send(client, "+256799000302", "orange powder under the leaves")
    assert TEMPLATES["welcome"]["en"] in r["replies"]
    assert TEMPLATES["adv_leaf_rust"]["en"] in r["replies"]


def test_a_luganda_keyword_switches_to_luganda(client):
    r = send(client, "+256799000303", "BBEEYI")
    assert any(m.startswith("Bbeeyi y'emmwanyi") for m in r["replies"])


def test_a_follow_up_digit_keeps_the_language(client):
    phone = "+256799000304"
    send(client, phone, "Ebikoola by'emmwanyi zange birina obubonero")
    client.post("/api/demo/clock", json={"phone": phone, "day": 0, "slot": "evening"})
    from app import db

    db.set_user(phone, pending_kind="followup")
    r = send(client, phone, "3")
    assert TEMPLATES["followup_worse"]["lg"] in r["replies"]


# ---------- prices ----------

@pytest.mark.parametrize("question", [
    "what is the price of coffee today?",
    "how much do they pay for parchment",
    "Emmwanyi zigula ssente mmeka leero?",
    "prix du café",
])
def test_a_price_question_in_free_text_gets_prices_without_ai(client, monkeypatch, question):
    import app.router

    monkeypatch.setattr(app.router, "analyze", _boom)
    r = send(client, "+256799000310", question)
    assert any("15,000-16,000" in m for m in r["replies"])
    assert "agent_alert" not in [e["kind"] for e in r["events"]]


def test_a_price_question_found_by_the_translation_gets_prices(client, monkeypatch):
    import app.router

    def translated(text, clarify_answer=None):
        return {"lang": "lg", "text_en": "How much does coffee sell for today?", "label": "other",
                "proba": 0.99, "llm_label": "other", "decision": "escalate",
                "template_id": "unsure", "reason": "other"}

    monkeypatch.setattr(app.router, "analyze", translated)
    r = send(client, "+256799000311", "Emmwanyi ziri ku ki leero")
    assert any("15,000-16,000" in m for m in r["replies"])
    assert "agent_alert" not in [e["kind"] for e in r["events"]]


def test_a_symptom_is_not_mistaken_for_a_price_question():
    from app.router import is_price_question

    assert not is_price_question("orange powder under the leaves")
    assert not is_price_question("help my coffee leaves have orange powder")
    assert is_price_question("How much is arabica parchment?")


# ---------- the clarifying question the LLM picks reaches Noor ----------

def test_the_question_chosen_by_the_chain_is_the_one_sent(client, monkeypatch):
    import app.router

    def unsure(text, clarify_answer=None):
        return {"lang": "en", "text_en": text, "label": "phoma", "proba": 0.5,
                "llm_label": "leaf_rust", "decision": "clarify",
                "template_id": "ask_powder_or_dark_patches", "reason": "disagree"}

    monkeypatch.setattr(app.router, "analyze", unsure)
    r = send(client, "+256799000320", "brown and orange marks on my leaves")
    assert TEMPLATES["ask_powder_or_dark_patches"]["en"] in r["replies"]


# ---------- Africa's Talking endpoint ----------

def test_sandbox_username_uses_the_sandbox(monkeypatch):
    from app import at_client

    monkeypatch.setenv("AT_USERNAME", "sandbox")
    assert "sandbox" in at_client.api_url()
    monkeypatch.setenv("AT_USERNAME", "coffee-coop")
    assert "sandbox" not in at_client.api_url()


# ---------- concurrency ----------

def test_two_messages_at_once_get_one_welcome(client, monkeypatch):
    """Back-to-back SMS reach the webhook together: one welcome, not two."""
    import threading
    import time

    import app.router
    from app import at_client, db

    def slow_gateway(to, text):
        time.sleep(0.2)  # the real HTTP call to Africa's Talking takes ~1 s

    monkeypatch.setattr(at_client, "send_sms", slow_gateway)
    phone = "+256799000400"
    threads = [threading.Thread(target=app.router.handle_incoming, args=(phone, t))
               for t in ("orange powder under the leaves", "small holes in the berries")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = [m["text"] for m in db.messages(phone) if m["direction"] == "out"]
    assert sum(m.startswith("Coffee Advisor:") for m in out) == 1, out
