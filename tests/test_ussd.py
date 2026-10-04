"""USSD menu: prices, report a symptom, ask for an agent.

Africa's Talking replays the whole session in `text`, segments joined by '*',
so the handler stays stateless. A reply starts with CON (keep the session) or
END (hang up).
"""

import pytest

P = "+256799000123"


def ussd(client, text="", phone=P):
    r = client.post(
        "/ussd",
        data={
            "sessionId": "sess-1",
            "serviceCode": "*384*12345#",
            "phoneNumber": phone,
            "text": text,
        },
    )
    assert r.status_code == 200
    return r.text


def test_root_menu_offers_the_three_entries(client):
    body = ussd(client)
    assert body.startswith("CON ")
    assert "1" in body and "2" in body and "3" in body
    assert "price" in body.lower() and "agent" in body.lower()


def test_entry_1_gives_prices_and_ends(client):
    body = ussd(client, "1")
    assert body.startswith("END ")
    assert "UGX" in body


def test_entry_2_asks_for_the_symptom(client):
    assert ussd(client, "2").startswith("CON ")


def test_entry_2_then_a_symptom_gives_the_advice(client):
    body = ussd(client, "2*my coffee leaves have orange powder")
    assert body.startswith("END ")
    assert "rust" in body.lower()


def test_entry_2_out_of_scope_hands_over_to_a_human(client, agent_sms):
    body = ussd(client, "2*my coffee berries are falling off")
    assert body.startswith("END ")
    assert "not sure" in body.lower()
    assert agent_sms


def test_entry_2_gibberish_ends_with_the_question(client, agent_sms):
    # The session closes, but the pending question survives: her SMS answer
    # completes the diagnosis.
    body = ussd(client, "2*zzzz qwerty")
    assert body.startswith("END ")
    assert "tell me more" in body
    assert not agent_sms


def test_entry_3_calls_a_human(client, agent_sms):
    body = ussd(client, "3")
    assert body.startswith("END ")
    assert agent_sms
    assert P in agent_sms[-1]["text"]


def test_a_ussd_session_leaves_the_same_trace_as_an_sms(client):
    ussd(client, "2*my coffee leaves have orange powder")
    st = client.get("/api/demo/state", params={"phone": P}).json()
    assert [m["label"] for m in st["messages"] if m["direction"] == "in"] == ["leaf_rust"]


def test_unknown_entry_shows_the_menu_again(client):
    assert ussd(client, "9").startswith("CON ")


@pytest.mark.parametrize("payload", [{}, {"phoneNumber": P}, {"text": "1"}, {"phoneNumber": "", "text": ""}])
def test_ussd_never_500s(client, payload):
    assert client.post("/ussd", data=payload).status_code == 200


def test_screens_fit_a_ussd_window(client):
    for text in ("", "1", "2", "3", "9"):
        assert len(ussd(client, text)) <= 182  # AT truncates past this
