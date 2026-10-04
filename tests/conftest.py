import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set before app.main is imported anywhere: the developer's own .env must not
# leak into the suite.
os.environ["SKIP_DOTENV"] = "1"

NOOR = "+256799000001"
OTHER = "+256799000002"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient on a throwaway database, with outbound SMS captured."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("AGENT_PHONE", "+256700000099")
    monkeypatch.setenv("USE_REAL_AI", "0")

    from fastapi.testclient import TestClient

    from app import db, main

    db.init()
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def agent_sms(monkeypatch):
    """Captures the SMS actually sent to AGENT_PHONE, not just the event."""
    from app import at_client

    sent = []
    real = at_client.send_sms

    def spy(to, text):
        if to == os.environ.get("AGENT_PHONE"):
            sent.append({"to": to, "text": text})
        return real(to, text)

    monkeypatch.setattr(at_client, "send_sms", spy)
    return sent
