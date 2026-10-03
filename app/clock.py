"""Time. Real for real phones, simulated per visitor for demo numbers.

Kampala is UTC+3 with no DST, so a fixed offset beats pulling tzdata onto Replit.
"""

import time
from datetime import datetime, timedelta, timezone

from app import db

TZ = timezone(timedelta(hours=3))
BASE = datetime(2026, 10, 1, tzinfo=TZ)  # demo day 0
SLOT_HOUR = {"morning": 8, "day": 13, "evening": 19, "night": 23}
DEMO_PREFIX = "+256799"
FOLLOWUP_HOUR = 18  # start of the 18:00-20:00 window


def is_demo(phone):
    return phone.startswith(DEMO_PREFIX)


def now(phone):
    """Epoch seconds on this phone's clock."""
    if not is_demo(phone):
        return int(time.time())
    row, _ = db.user(phone)
    return slot_ts(row["day"], row["slot"])


def slot_ts(day, slot):
    return int((BASE + timedelta(days=day, hours=SLOT_HOUR[slot])).timestamp())


def followup_due(ts):
    """Three days after a diagnosis, at the start of the evening window.

    Due means now >= this instant, never "inside the window": a demo clock that
    jumps from day 0 to day 5 must still deliver the follow-up, not swallow it.
    """
    d = datetime.fromtimestamp(ts, TZ) + timedelta(days=3)
    return int(d.replace(hour=FOLLOWUP_HOUR, minute=0, second=0, microsecond=0).timestamp())


def iso(ts):
    return datetime.fromtimestamp(ts, TZ).isoformat()
