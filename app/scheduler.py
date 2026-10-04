"""Follow-up jobs, three days after a diagnosis.

run_due() is called both from the background loop and from every demo request
(Replit puts the instance to sleep), so claiming a job has to be atomic.
"""

import logging

from app import clock, content, db

log = logging.getLogger(__name__)


def schedule(phone, kind, due_ts):
    with db.lock:
        c = db.connect()
        c.execute(
            "INSERT INTO jobs(phone, kind, due_ts) VALUES(?,?,?)", (phone, kind, due_ts)
        )
        c.commit()


def cancel(phone):
    with db.lock:
        c = db.connect()
        c.execute("UPDATE jobs SET done=1 WHERE phone=? AND done=0", (phone,))
        c.commit()


def _claim(job_id):
    """True for exactly one caller, however many race for the same job."""
    with db.lock:
        c = db.connect()
        cur = c.execute("UPDATE jobs SET done=1 WHERE id=? AND done=0", (job_id,))
        c.commit()
        return cur.rowcount == 1


def run_due(phone=None):
    sql = "SELECT id, phone, kind, due_ts FROM jobs WHERE done=0"
    args = ()
    if phone:
        sql += " AND phone=?"
        args = (phone,)
    for job in db.connect().execute(sql, args).fetchall():
        # Each phone has its own clock, so the comparison is per row.
        now = clock.now(job["phone"])
        if now >= job["due_ts"] and _claim(job["id"]):
            try:
                _fire(job, now)
            except Exception:
                log.exception("follow-up job %s failed", job["id"])


def _fire(job, now):
    phone = job["phone"]
    row, _ = db.user(phone)
    if not db.send(phone, content.t("followup_q", row["lang"]), now):
        return  # muted by STOP
    db.set_user(phone, pending_kind="followup", pending_text=None)
    db.event(phone, "followup", "follow-up sent", now)
