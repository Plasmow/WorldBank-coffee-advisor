"""The English subtitles on the demo page (judges only).

Our replies use the template's own English; Noor's messages use Google
Translate, or the chain's own translation when Google is unavailable.
"""

import time

import httpx
import pytest

from tests.conftest import NOOR


@pytest.fixture(autouse=True)
def fresh_google():
    from app import gtranslate

    gtranslate._cache.clear()
    gtranslate._blocked_until = 0.0
    yield
    gtranslate._cache.clear()
    gtranslate._blocked_until = 0.0


def english(client, phone):
    r = client.get("/api/demo/english", params={"phone": phone})
    assert r.status_code == 200, r.text
    return r.json()["messages"]


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.request = httpx.Request("GET", "https://x")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=self.request, response=self)

    def json(self):
        return self._body


def test_replies_use_the_template_english_and_never_call_google(client, monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(k) or Resp(
        body=[[["Under the leaves there is orange powder", "x", None]], None, "lg"]))
    client.post("/api/demo/send", json={"phone": NOOR, "text": "Obuwunga bwa kacungwa wansi w'ebikoola"})
    client.post("/api/demo/send", json={"phone": NOOR, "text": "BBEEYI"})

    rows = english(client, NOOR)
    replies = [r for r in rows if r["direction"] == "out"]
    assert replies and all(r["via"] == "template" for r in replies)
    assert any(r["en"].startswith("Coffee prices") for r in replies)       # Luganda price message
    assert any(r["en"].startswith("Coffee Advisor: free") for r in replies)  # Luganda welcome
    # Google was asked only about Noor's two messages, never about a reply.
    assert len(calls) <= 2


def test_noor_message_falls_back_to_our_model_when_google_is_down(client, monkeypatch):
    import app.router

    def chain(text, clarify_answer=None):
        return {"lang": "lg", "text_en": "orange powder under the leaves", "label": "leaf_rust",
                "proba": 0.95, "llm_label": "leaf_rust", "decision": "answer",
                "template_id": "adv_leaf_rust", "reason": "agree"}

    monkeypatch.setattr(app.router, "analyze", chain)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(429))
    client.post("/api/demo/send", json={"phone": NOOR, "text": "Obuwunga bwa kacungwa wansi w'ebikoola"})

    noor = [r for r in english(client, NOOR) if r["direction"] == "in"][0]
    assert (noor["en"], noor["via"]) == ("orange powder under the leaves", "opus")


def test_a_429_pauses_google_instead_of_hammering_it(monkeypatch):
    from app import gtranslate

    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(1) or Resp(429, headers={"Retry-After": "120"}))
    for _ in range(5):
        assert gtranslate.to_english(["Oli otya ssebo"]) == [(None, None)]
    assert len(calls) == 1                                   # one request, then silence
    assert gtranslate._blocked_until > time.monotonic() + 100  # Retry-After honoured


def test_google_is_asked_again_after_the_cooldown(monkeypatch):
    from app import gtranslate

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(body=[[["How are you sir", "x", None]], None, "lg"]))
    gtranslate._blocked_until = time.monotonic() - 1
    assert gtranslate.to_english(["Oli otya ssebo"]) == [("How are you sir", "lg")]


def test_english_route_refuses_real_numbers(client):
    assert client.get("/api/demo/english", params={"phone": "+256701234567"}).status_code == 400


def test_digits_are_not_sent_to_google(monkeypatch):
    from app import gtranslate

    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert gtranslate.to_english(["3"]) == [("3", None)]


def test_free_endpoint_response_is_parsed(monkeypatch):
    from app import gtranslate

    body = [[["My coffee ", "Emmwanyi zange ", None], ["looks bad", "zirabika bubi", None]], None, "lg"]
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(body=body))
    assert gtranslate.to_english(["Emmwanyi zange zirabika bubi"]) == [("My coffee looks bad", "lg")]


def test_english_text_is_shown_as_is_not_sent_to_google(monkeypatch):
    from app import content, gtranslate

    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    prices = content.prices_text("en")
    assert gtranslate.to_english([prices]) == [(prices, "en")]


def test_every_template_maps_back_to_its_english():
    import json
    import pathlib

    from app import content

    templates = json.loads((pathlib.Path(__file__).parent.parent / "data" / "templates.json")
                           .read_text(encoding="utf-8"))
    for key, v in templates.items():
        if "{" in v["en"]:
            continue  # the price wrapper, covered above
        assert content.english_of(v["lg"]) == v["en"], key
