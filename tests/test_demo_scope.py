"""The demo endpoints are public on Replit. They must only ever touch demo
numbers.

Two things go wrong otherwise: anyone can read a real farmer's whole
conversation by guessing her number, and anyone can make the server send SMS
on our Africa's Talking credit.
"""

NOOR_DEMO = "+256799000001"
REAL = "+256701234567"


def test_a_real_number_cannot_be_driven_through_the_demo_endpoint(client):
    r = client.post("/api/demo/send", json={"phone": REAL, "text": "hello"})
    assert r.status_code == 400


def test_a_real_conversation_cannot_be_read_through_the_demo_endpoint(client):
    # Seed a real conversation the way the SMS webhook would.
    client.post("/sms", data={"from": REAL, "text": "orange powder", "id": "x1"})

    r = client.get("/api/demo/state", params={"phone": REAL})
    assert r.status_code == 400
    assert "orange powder" not in r.text


def test_a_real_number_cannot_have_its_clock_moved(client):
    assert client.post("/api/demo/clock", json={"phone": REAL, "day": 3, "slot": "evening"}).status_code == 400


def test_a_real_number_cannot_be_wiped(client):
    assert client.post("/api/demo/reset", json={"phone": REAL}).status_code == 400


def test_demo_numbers_still_work(client):
    assert client.post("/api/demo/send", json={"phone": NOOR_DEMO, "text": "PRICE"}).status_code == 200
    assert client.get("/api/demo/state", params={"phone": NOOR_DEMO}).status_code == 200


# --- live mode: one real number, chosen by the operator ------------------


def test_the_live_number_is_accepted_when_the_operator_set_one(client, monkeypatch):
    monkeypatch.setenv("DEMO_LIVE_PHONE", REAL)
    r = client.post("/api/demo/send", json={"phone": REAL, "text": "PRICE"})
    assert r.status_code == 200


def test_the_same_number_is_refused_when_no_live_number_is_set(client, monkeypatch):
    monkeypatch.delenv("DEMO_LIVE_PHONE", raising=False)
    assert client.post("/api/demo/send", json={"phone": REAL, "text": "PRICE"}).status_code == 400


def test_live_mode_opens_one_number_not_every_number(client, monkeypatch):
    monkeypatch.setenv("DEMO_LIVE_PHONE", REAL)
    other = "+256709999999"
    assert client.post("/api/demo/send", json={"phone": other, "text": "PRICE"}).status_code == 400


def test_the_live_number_is_matched_whatever_the_format(client, monkeypatch):
    monkeypatch.setenv("DEMO_LIVE_PHONE", "0701234567")  # same farmer, local spelling
    assert client.post("/api/demo/send", json={"phone": REAL, "text": "PRICE"}).status_code == 200


def test_config_tells_the_page_what_is_available(client, monkeypatch):
    monkeypatch.delenv("DEMO_LIVE_PHONE", raising=False)
    assert client.get("/api/demo/config").json()["live_phone"] is None

    monkeypatch.setenv("DEMO_LIVE_PHONE", REAL)
    body = client.get("/api/demo/config").json()
    assert body["live_phone"] == REAL
    assert body["demo_mode"] is True  # DEMO_MODE=1 in the fixtures: nothing would leave
