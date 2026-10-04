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
