import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
