"""Where configuration comes from, and who wins.

Replit passes its Secrets as real environment variables. A .env file left
lying around from a laptop must never be able to override them, or a stale
sandbox key quietly replaces the production one.
"""

import os


def test_the_file_fills_gaps_but_never_overrides_the_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AT_API_KEY=from-the-file\n"
        "AT_SHORTCODE=6789\n"
    )
    monkeypatch.setenv("AT_API_KEY", "from-host-secrets")
    monkeypatch.delenv("AT_SHORTCODE", raising=False)

    from app.main import load_env

    load_env(env_file)

    assert os.environ["AT_API_KEY"] == "from-host-secrets"  # the real one wins
    assert os.environ["AT_SHORTCODE"] == "6789"               # the file fills the gap


def test_a_missing_file_is_not_an_error(tmp_path):
    from app.main import load_env

    load_env(tmp_path / "nothing-here")  # must not raise


def test_a_rejected_sms_is_logged_loudly(monkeypatch, caplog):
    """A 401 from the gateway must not look like a successful send."""
    import logging
    import httpx

    from app import at_client

    class Rejected:
        status_code = 401
        text = '{"message":"Invalid API key"}'

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Rejected())

    with caplog.at_level(logging.ERROR, logger="app.at_client"):
        at_client.send_sms("+256701234567", "hello")

    assert any("401" in r.getMessage() for r in caplog.records)
    assert any("Invalid API key" in r.getMessage() for r in caplog.records)


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code, self.text = status_code, text

    def json(self):
        import json

        return json.loads(self.text)


SENT = '{"SMSMessageData":{"Message":"Sent to 1/1","Recipients":[{"messageId":"ATXid_1"}]}}'
REFUSED = '{"SMSMessageData":{"Message":"InvalidSenderId","Recipients":[]}}'


def _capture_post(monkeypatch):
    """Intercepts the call to the gateway and hands back what was sent."""
    import httpx

    from app import at_client

    seen = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        seen.update(data or {})
        return FakeResponse(201, SENT)

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(httpx, "post", fake_post)
    return at_client, seen


def test_an_empty_shortcode_is_left_out_of_the_request(monkeypatch):
    # The sandbox answers InvalidSenderId to an empty sender id, and accepts
    # the message only when the field is absent altogether.
    at_client, seen = _capture_post(monkeypatch)
    monkeypatch.setenv("AT_SHORTCODE", "")

    at_client.send_sms("+256700000099", "hello")

    assert "from" not in seen


def test_a_real_shortcode_is_sent(monkeypatch):
    at_client, seen = _capture_post(monkeypatch)
    monkeypatch.setenv("AT_SHORTCODE", "12345")

    at_client.send_sms("+256700000099", "hello")

    assert seen["from"] == "12345"


def test_a_refusal_dressed_as_201_is_logged(monkeypatch, caplog):
    """Africa's Talking answers 201 and puts the failure in the body."""
    import logging

    import httpx

    from app import at_client

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(201, REFUSED))

    with caplog.at_level(logging.ERROR, logger="app.at_client"):
        at_client.send_sms("+256700000099", "hello")

    assert any("InvalidSenderId" in r.getMessage() for r in caplog.records)


def test_a_delivered_message_is_not_logged_as_an_error(monkeypatch, caplog):
    import logging

    import httpx

    from app import at_client

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(201, SENT))

    with caplog.at_level(logging.ERROR, logger="app.at_client"):
        at_client.send_sms("+256700000099", "hello")

    assert not caplog.records
