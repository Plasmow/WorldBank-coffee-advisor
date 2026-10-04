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
    monkeypatch.setenv("AT_API_KEY", "from-replit-secrets")
    monkeypatch.delenv("AT_SHORTCODE", raising=False)

    from app.main import load_env

    load_env(env_file)

    assert os.environ["AT_API_KEY"] == "from-replit-secrets"  # the real one wins
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
