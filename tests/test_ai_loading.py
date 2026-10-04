"""Loading the real chain must never be able to take the server down.

A hackathon demo fails in one of two ways: the model is missing, or it is
there and something inside it throws. Both must end with a farmer talking to
a human, never with a boot crash or a silent non-answer.
"""

import sys
import types

import pytest


def fake_ai_module(fn):
    """Stands in for ai.analyze without importing anything heavy."""
    mod = types.ModuleType("ai.analyze")
    mod.analyze = fn
    return mod


def test_health_reports_memory_and_ai_state(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["rss_mb"] > 0
    assert body["ai"]["enabled"] is False  # USE_REAL_AI=0 in the fixtures


def test_preload_does_nothing_and_imports_nothing_when_disabled(client, monkeypatch):
    from app import analysis

    monkeypatch.setenv("USE_REAL_AI", "0")
    monkeypatch.delitem(sys.modules, "ai.translate", raising=False)
    assert analysis.preload()["enabled"] is False
    assert "ai.translate" not in sys.modules  # nothing heavy was touched


def test_the_server_still_boots_when_the_model_is_missing(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("USE_REAL_AI", "1")
    monkeypatch.setenv("NLLB_CT2_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("NLLB_CT2_REPO", "")

    from app import main

    with TestClient(main.app) as c:          # boots, does not raise
        body = c.get("/health").json()
    assert body["ok"] is True
    assert body["ai"]["enabled"] is True
    assert body["ai"]["loaded"] is False     # and says so plainly


def test_a_chain_that_throws_sends_the_farmer_to_a_human(client, monkeypatch, agent_sms):
    def boom(text, clarify_answer=None):
        raise RuntimeError("CUDA is on fire")

    monkeypatch.setenv("USE_REAL_AI", "1")
    monkeypatch.setitem(sys.modules, "ai.analyze", fake_ai_module(boom))

    r = client.post(
        "/api/demo/send", json={"phone": "+256799000055", "text": "my leaves are odd"}
    ).json()
    assert any("not sure" in m.lower() for m in r["replies"])
    assert "agent_alert" in [e["kind"] for e in r["events"]]
    assert agent_sms


def test_a_chain_returning_junk_sends_the_farmer_to_a_human(client, monkeypatch, agent_sms):
    # P3's stub currently returns empty strings for every field. Without a
    # check that lands as a reply with no template and no agent alerted.
    def empty(text, clarify_answer=None):
        return {"lang": "", "text_en": "", "label": "", "proba": "",
                "llm_label": "", "decision": "", "template_id": "", "reason": ""}

    monkeypatch.setenv("USE_REAL_AI", "1")
    monkeypatch.setitem(sys.modules, "ai.analyze", fake_ai_module(empty))

    r = client.post(
        "/api/demo/send", json={"phone": "+256799000056", "text": "my leaves are odd"}
    ).json()
    assert any("not sure" in m.lower() for m in r["replies"])
    assert "agent_alert" in [e["kind"] for e in r["events"]]


def test_a_well_formed_chain_result_is_passed_through(client, monkeypatch):
    def good(text, clarify_answer=None):
        return {"lang": "lg", "text_en": "orange powder", "label": "leaf_rust",
                "proba": 0.93, "llm_label": "leaf_rust", "decision": "answer",
                "template_id": "adv_leaf_rust", "reason": ""}

    monkeypatch.setenv("USE_REAL_AI", "1")
    monkeypatch.setitem(sys.modules, "ai.analyze", fake_ai_module(good))

    r = client.post(
        "/api/demo/send", json={"phone": "+256799000057", "text": "ebikoola birimu"}
    ).json()
    assert any("rust" in m.lower() for m in r["replies"])
    incoming = [m for m in r["messages"] if m["direction"] == "in"][-1]
    assert incoming["label"] == "leaf_rust" and incoming["proba"] == 0.93


def test_translate_says_plainly_what_is_missing(monkeypatch, tmp_path):
    import ai.translate as tr

    monkeypatch.setenv("NLLB_CT2_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("NLLB_CT2_REPO", "")
    tr.get_translator.cache_clear()
    with pytest.raises(FileNotFoundError, match="convert_nllb_ct2"):
        tr.load()
