"""Control B: the internal acceptance run, one number from first contact to
the human safety net. Flip USE_REAL_AI=1 to run the same script through P3's
chain once it lands -- the assertions are about the journey, not the model.
"""

NOOR = "+256799000777"


def post(client, path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_control_b_full_acceptance_run(client, agent_sms):
    # 1. first contact: welcome, then a diagnosis
    st = post(client, "/api/demo/send", phone=NOOR, text="my coffee leaves have orange powder underneath")
    assert any("Coffee Advisor" in m for m in st["replies"])
    assert any("rust" in m.lower() for m in st["replies"])

    # 2. the inbound message carries its analysis
    first_in = [m for m in st["messages"] if m["direction"] == "in"][0]
    assert first_in["label"] == "leaf_rust"

    # 3. market prices, with no AI in the loop
    st = post(client, "/api/demo/send", phone=NOOR, text="PRICE")
    assert any("UGX" in m for m in st["replies"])

    # 4. three days later the follow-up goes out on its own
    st = post(client, "/api/demo/clock", phone=NOOR, day=3, slot="evening")
    assert "followup" in [e["kind"] for e in st["events"]]

    # 5. "worse" hands the farmer to a human
    st = post(client, "/api/demo/send", phone=NOOR, text="3")
    assert "agent_alert" in [e["kind"] for e in st["events"]]
    assert agent_sms, "the agent must receive a real SMS, not just an event"

    # 6. STOP is honoured and nothing more is ever sent
    st = post(client, "/api/demo/send", phone=NOOR, text="STOP")
    before = len([m for m in st["messages"] if m["direction"] == "out"])
    st = post(client, "/api/demo/send", phone=NOOR, text="my leaves have orange powder")
    assert len([m for m in st["messages"] if m["direction"] == "out"]) == before
