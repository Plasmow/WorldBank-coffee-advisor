"""One entry point for every inbound message, whatever the channel."""

import logging
import os
import unicodedata

from app import at_client, clock, content, db, scheduler
from app.analysis import analyze

log = logging.getLogger(__name__)

MAX_IN = 1000  # a gateway can hand us anything; the classifier does not need more

# Both spellings accepted: the spec is written in French but Noor's handset
# sends English. Two extra strings beat getting this wrong on stage.
KEYWORDS = {
    "PRICE": ("price", "prices", "prix"),
    "HELP": ("help", "aide"),
    "STOP": ("stop",),
    "START": ("start",),
}
FOLLOWUP_ANSWERS = {"1": "better", "2": "same", "3": "worse"}


def keyword(text):
    """Exact match on the whole message, so "help my coffee is dying" is a
    symptom report and not the HELP menu."""
    w = unicodedata.normalize("NFKD", text.strip().lower())
    w = w.encode("ascii", "ignore").decode()
    for name, words in KEYWORDS.items():
        if w in words:
            return name
    return None


def handle_incoming(phone, text):
    """Returns the messages actually sent to this phone."""
    phone = db.norm_phone(phone)
    text = (text or "").strip()[:MAX_IN]
    if not phone or not text:
        return []

    ts = clock.now(phone)
    row, is_new = db.user(phone)
    db.record(phone, "in", text, ts)

    sent = []

    def say(key_or_text, literal=False):
        msg = key_or_text if literal else content.t(key_or_text, row["lang"])
        if msg and db.send(phone, msg, ts):
            sent.append(msg)

    kw = keyword(text)

    if kw == "STOP":
        # Acknowledge first, then mute: the ack itself must not be swallowed.
        db.send(phone, content.t("stop_ack", row["lang"]), ts, force=True)
        db.set_user(phone, stopped=1, pending_kind=None, pending_text=None)
        scheduler.cancel(phone)
        return [content.t("stop_ack", row["lang"])]

    if kw == "START":
        db.set_user(phone, stopped=0)
        say("start_ack")
        return sent

    if row["stopped"]:
        return []  # recorded for the trace, never answered

    if is_new:
        say("welcome")

    if kw == "PRICE":
        say(content.prices_text(row["lang"]), literal=True)  # no AI on this path
        return sent

    if kw == "HELP":
        say("help")
        return sent

    if row["pending_kind"] == "followup":
        db.set_user(phone, pending_kind=None)
        answer = FOLLOWUP_ANSWERS.get(text.strip())
        if answer == "worse":
            say("followup_worse")
            _alert_agent(phone, ts, "follow-up: getting worse")
            return sent
        if answer:
            say("followup_ok")
            return sent
        # Not 1/2/3: treat it as a fresh report rather than nagging.

    if row["pending_kind"] == "clarify":
        db.set_user(phone, pending_kind=None, pending_text=None)
        _respond(phone, ts, row, say, analyze(row["pending_text"], clarify_answer=text))
        return sent

    _respond(phone, ts, row, say, analyze(text))
    return sent


def _respond(phone, ts, row, say, result):
    decision = result.get("decision")

    if decision == "clarify":
        db.set_user(phone, pending_kind="clarify", pending_text=result.get("text_en") or "")
        say("clarify")
        return

    if decision == "escalate":
        say("unsure")
        _alert_agent(phone, ts, f"unclear report: {result.get('text_en', '')[:120]}")
        return

    say(result.get("template_id") or "Unknown", literal=False)
    if result.get("label") in ("leaf_rust", "phoma"):
        scheduler.schedule(phone, "followup", clock.followup_due(ts))


def _alert_agent(phone, ts, reason):
    """The human safety net. Recorded as an event so the demo page can show it."""
    text = f"Coffee Advisor: {phone} needs a human. {reason}"
    db.event(phone, "agent_alert", text, ts)
    agent = os.environ.get("AGENT_PHONE", "")
    if agent:
        at_client.send_sms(agent, text)
