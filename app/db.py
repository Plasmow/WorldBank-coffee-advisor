"""SQLite storage and the single outbound path to the farmer.

Every message sent to Noor goes through send(): the STOP check lives there, so
no caller -- router, scheduler, a future retry -- can bypass it.
"""

import os
import re
import sqlite3
import threading

from app import at_client

# ponytail: one global lock. Replit runs a single worker; swap for per-phone
# locks only if a second instance ever appears.
lock = threading.Lock()
_conn = None
_conn_path = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  phone        TEXT PRIMARY KEY,
  stopped      INTEGER NOT NULL DEFAULT 0,
  lang         TEXT    NOT NULL DEFAULT 'en',
  pending_kind TEXT,
  pending_text TEXT,
  day          INTEGER NOT NULL DEFAULT 0,
  slot         TEXT    NOT NULL DEFAULT 'morning'
);
CREATE TABLE IF NOT EXISTS messages(
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  phone     TEXT NOT NULL,
  direction TEXT NOT NULL,
  text      TEXT NOT NULL,
  ts        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  phone TEXT NOT NULL,
  kind  TEXT NOT NULL,
  text  TEXT NOT NULL,
  ts    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs(
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  phone  TEXT NOT NULL,
  kind   TEXT NOT NULL,
  due_ts INTEGER NOT NULL,
  done   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS seen(at_id TEXT PRIMARY KEY);
"""


def connect():
    global _conn, _conn_path
    path = os.environ.get("DB_PATH", "data/app.db")
    if _conn is None or _conn_path != path:
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
        _conn_path = path
    return _conn


def init():
    connect()


def norm_phone(raw):
    """0799.. / 256799.. / +256799.. all collapse to one key.

    Without this the same farmer gets several rows, STOP leaks and two demo
    visitors can see each other.
    """
    p = re.sub(r"[^\d+]", "", raw or "")
    if p.startswith("00"):
        p = "+" + p[2:]
    elif p.startswith("0"):
        p = "+256" + p[1:]
    elif p and not p.startswith("+"):
        p = "+" + p
    return p


def get_user(phone):
    """Read-only: never creates a row, so callers like clock.now() cannot
    silently consume a farmer's first contact."""
    return connect().execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()


def user(phone):
    """Return (row, is_new), creating the row on first contact."""
    with lock:
        c = connect()
        row = c.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()
        if row:
            return row, False
        c.execute("INSERT INTO users(phone) VALUES(?)", (phone,))
        c.commit()
        row = c.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()
        return row, True


def set_user(phone, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    with lock:
        c = connect()
        c.execute(f"UPDATE users SET {cols} WHERE phone=?", (*fields.values(), phone))
        c.commit()


def record(phone, direction, text, ts):
    with lock:
        c = connect()
        c.execute(
            "INSERT INTO messages(phone, direction, text, ts) VALUES(?,?,?,?)",
            (phone, direction, text, ts),
        )
        c.commit()


def event(phone, kind, text, ts):
    with lock:
        c = connect()
        c.execute(
            "INSERT INTO events(phone, kind, text, ts) VALUES(?,?,?,?)",
            (phone, kind, text, ts),
        )
        c.commit()


def send(phone, text, ts, force=False):
    """The only way a message reaches the farmer. Returns False if dropped.

    force is for the STOP acknowledgement itself, which must go out even
    though the user is being muted in the same breath.
    """
    row, _ = user(phone)
    if row["stopped"] and not force:
        return False
    record(phone, "out", text, ts)
    at_client.send_sms(phone, text)
    return True


def seen_once(at_id):
    """False if this gateway message id was already handled (AT retries)."""
    if not at_id:
        return True
    with lock:
        c = connect()
        try:
            c.execute("INSERT INTO seen(at_id) VALUES(?)", (at_id,))
            c.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def messages(phone):
    # Ordered by id, not ts: two messages in the same demo slot share a timestamp.
    return connect().execute(
        "SELECT id, direction, text, ts FROM messages WHERE phone=? ORDER BY id", (phone,)
    ).fetchall()


def events(phone):
    return connect().execute(
        "SELECT id, kind, text, ts FROM events WHERE phone=? ORDER BY id", (phone,)
    ).fetchall()


def reset(phone):
    with lock:
        c = connect()
        for table in ("messages", "events", "jobs", "users"):
            c.execute(f"DELETE FROM {table} WHERE phone=?", (phone,))
        c.commit()
