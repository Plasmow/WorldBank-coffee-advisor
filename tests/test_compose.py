"""Typing English in the demo page and sending Luganda to the backend.

Whoever drives the demo does not speak Luganda, but the backend must receive
what a farmer would really send. The page translates on the way in; nothing
here touches the AI chain, which still sees Luganda like any other SMS.
"""

import pytest

DEMO = "+256799000042"


@pytest.fixture
def no_network(monkeypatch):
    """Google is never called in the tests."""
    import httpx

    from app import gtranslate

    gtranslate._cache.clear()
    gtranslate._blocked_until = 0.0

    def fake_get(url, params=None, timeout=None):
        assert params["tl"] == "lg", "the composer must ask for Luganda"
        return _Resp([[[f"[lg] {params['q']}", params["q"]]], None, "en"])

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    return gtranslate


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_english_in_luganda_out(client, no_network):
    r = client.post("/api/demo/compose", json={"text": "orange powder under the leaves"})
    assert r.status_code == 200
    assert r.json()["lg"] == "[lg] orange powder under the leaves"


def test_luganda_in_is_left_alone(client, no_network):
    # Already Luganda: translating it again would only damage it.
    text = "Ebikoola by'emmwanyi zange birina obuwunga"
    assert client.post("/api/demo/compose", json={"text": text}).json()["lg"] == text


def test_an_empty_message_is_refused(client, no_network):
    assert client.post("/api/demo/compose", json={"text": "   "}).status_code == 400


def test_a_very_long_message_is_refused(client, no_network):
    assert client.post("/api/demo/compose", json={"text": "x" * 5000}).status_code == 400


def test_google_down_says_so_instead_of_sending_english(client, monkeypatch):
    import httpx

    from app import gtranslate

    gtranslate._cache.clear()
    gtranslate._blocked_until = 0.0
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    body = client.post("/api/demo/compose", json={"text": "orange powder"}).json()
    assert body["lg"] is None      # the page must not quietly send English
    assert body["via"] is None


def test_the_composer_does_not_touch_the_conversation(client, no_network):
    client.post("/api/demo/compose", json={"text": "orange powder under the leaves"})
    state = client.get("/api/demo/state", params={"phone": DEMO}).json()
    assert state["messages"] == []   # composing is not sending
