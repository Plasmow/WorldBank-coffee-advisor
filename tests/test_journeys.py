"""The eight journeys the demo has to survive."""

import json
import pathlib

import pytest

NOOR = "+256799000001"
OTHER = "+256799000002"
RUST = "my coffee leaves have orange powder underneath"


def send(client, phone, text):
    r = client.post("/api/demo/send", json={"phone": phone, "text": text})
    assert r.status_code == 200
    return r.json()


def state(client, phone):
    r = client.get("/api/demo/state", params={"phone": phone})
    assert r.status_code == 200
    return r.json()


def kinds(st):
    return [e["kind"] for e in st["events"]]


def outbound(st):
    return [m["text"] for m in st["messages"] if m["direction"] == "out"]


def inbound(st):
    return [m["text"] for m in st["messages"] if m["direction"] == "in"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# 1. conseil direct
def test_direct_advice(client):
    r = send(client, NOOR, RUST)
    assert any("rust" in m.lower() for m in r["replies"])
    assert any("Coffee Advisor" in m for m in r["replies"])  # welcome on first contact


# 2. question puis conseil
def test_clarify_then_advice(client):
    first = send(client, NOOR, "my coffee plants look bad")
    assert any("are the marks" in m for m in first["replies"])

    second = send(client, NOOR, "orange powder on the leaves")
    assert any("rust" in m.lower() for m in second["replies"])

    # exactly one clarifying question in the whole conversation
    assert sum("are the marks" in m for m in outbound(second)) == 1


# 3. transmission a l'agent
def test_escalate_to_agent(client):
    r = send(client, NOOR, "zzzz qwerty")
    assert any("not sure" in m.lower() for m in r["replies"])
    assert "agent_alert" in kinds(r)


def test_escalate_when_clarification_does_not_help(client):
    send(client, NOOR, "my coffee plants look bad")
    r = send(client, NOOR, "i cannot tell")
    assert any("not sure" in m.lower() for m in r["replies"])
    assert "agent_alert" in kinds(r)


# 4. PRIX
def test_price_keyword_never_calls_the_ai(client, monkeypatch):
    import app.router

    monkeypatch.setattr(app.router, "analyze", _boom)
    r = send(client, NOOR, "PRICE")
    assert any("UGX" in m for m in r["replies"])


def _boom(*a, **k):
    raise AssertionError("PRICE must not reach the AI chain")


# 5. AIDE
def test_help_keyword(client):
    r = send(client, NOOR, "help")
    assert any("PRICE" in m for m in r["replies"])


def test_help_inside_a_sentence_is_a_symptom_not_the_menu(client):
    r = send(client, NOOR, "help my coffee leaves have orange powder")
    assert any("rust" in m.lower() for m in r["replies"])


# 6. STOP / START
def test_stop_then_start(client):
    send(client, NOOR, RUST)
    stopped = send(client, NOOR, "STOP")
    assert any("no more messages" in m.lower() for m in stopped["replies"])

    silent = send(client, NOOR, RUST)
    assert silent["replies"] == []

    # a scheduled follow-up must not survive STOP
    client.post("/api/demo/clock", json={"phone": NOOR, "day": 3, "slot": "evening"})
    assert "followup" not in kinds(state(client, NOOR))

    back = send(client, NOOR, "START")
    assert any("welcome back" in m.lower() for m in back["replies"])
    assert any("rust" in m.lower() for m in send(client, NOOR, RUST)["replies"])


# 7. suivi J+3
def test_followup_day3(client):
    send(client, NOOR, RUST)
    st = state(client, NOOR)
    assert "followup" not in kinds(st)  # not yet

    st = client.post(
        "/api/demo/clock", json={"phone": NOOR, "day": 3, "slot": "evening"}
    ).json()
    assert "followup" in kinds(st)
    assert any("3 days on" in m for m in outbound(st))

    worse = send(client, NOOR, "3")
    assert "agent_alert" in kinds(worse)


def test_followup_fires_even_if_the_clock_jumps_past_the_window(client):
    send(client, NOOR, RUST)
    st = client.post(
        "/api/demo/clock", json={"phone": NOOR, "day": 6, "slot": "night"}
    ).json()
    assert "followup" in kinds(st)


def test_followup_sent_once_even_when_run_due_runs_twice(client):
    send(client, NOOR, RUST)
    client.post("/api/demo/clock", json={"phone": NOOR, "day": 3, "slot": "evening"})
    st = state(client, NOOR)
    assert kinds(st).count("followup") == 1


# 8. isolation de deux numeros
def test_two_demo_numbers_are_isolated(client):
    send(client, NOOR, RUST)
    send(client, OTHER, "PRICE")

    noor, other = state(client, NOOR), state(client, OTHER)
    assert inbound(noor) == [RUST]
    assert inbound(other) == ["PRICE"]

    client.post("/api/demo/clock", json={"phone": NOOR, "day": 5, "slot": "night"})
    assert state(client, NOOR)["now"] != state(client, OTHER)["now"]


def test_stop_is_per_number(client):
    send(client, NOOR, "STOP")
    assert send(client, OTHER, RUST)["replies"] != []


# robustesse du webhook
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"from": "+256700000001"},
        {"text": "hello"},
        {"from": "", "text": ""},
        {"from": "+256700000001", "text": "   "},
        {"from": "+256700000001", "text": "x" * 2000},
        {"from": "not a phone", "text": "hello"},
    ],
)
def test_webhook_never_500s(client, payload):
    assert client.post("/sms", data=payload).status_code == 200


def test_webhook_answers_a_normal_sms(client):
    r = client.post("/sms", data={"from": NOOR, "text": RUST, "id": "at-1"})
    assert r.status_code == 200
    assert any("rust" in m.lower() for m in outbound(state(client, NOOR)))


def test_webhook_ignores_a_replayed_message(client):
    for _ in range(3):
        client.post("/sms", data={"from": NOOR, "text": RUST, "id": "at-same"})
    inbound = [m for m in state(client, NOOR)["messages"] if m["direction"] == "in"]
    assert len(inbound) == 1


def test_webhook_accepts_local_phone_format(client):
    client.post("/sms", data={"from": "0799000001", "text": RUST, "id": "at-2"})
    assert outbound(state(client, NOOR))  # same farmer as +256799000001


def test_templates_fit_in_one_sms(client):
    tpl = json.loads(pathlib.Path("data/templates.json").read_text())
    too_long = {k: len(v) for k, v in tpl.items() if isinstance(v, str) and len(v) > 160}
    assert not too_long
