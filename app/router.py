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
    "AGENT": ("agent",),
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
    row, _ = db.user(phone)
    message_id = db.record(phone, "in", text, ts)

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

    # Explicit flag, not "the row did not exist": /api/demo/clock creates the
    # row before Noor has typed anything, and that must not eat her welcome.
    if not row["welcomed"]:
        say("welcome")
        db.set_user(phone, welcomed=1)

    if kw == "PRICE":
        say(content.prices_text(row["lang"]), literal=True)  # no AI on this path
        return sent

    if kw == "HELP":
        say("help")
        alert_agent(phone, ts, text, note="asked for HELP")
        return sent

    if kw == "AGENT":
        say("agent_requested")
        alert_agent(phone, ts, text, note="asked for an agent")
        return sent

    if row["pending_kind"] == "followup":
        db.set_user(phone, pending_kind=None)
        answer = FOLLOWUP_ANSWERS.get(text.strip())
        if answer == "worse":
            say("followup_worse")
            alert_agent(phone, ts, text, note="follow-up: getting worse")
            return sent
        if answer:
            say("followup_ok")
            return sent
        # Not 1/2/3: treat it as a fresh report rather than nagging.

    if row["pending_kind"] == "clarify":
        db.set_user(phone, pending_kind=None, pending_text=None)
        original = row["pending_text"] or text
        result = analyze(original, clarify_answer=text)
        db.annotate(message_id, result)
        _respond(phone, ts, say, result, original)
        return sent

    result = analyze(text)
    db.annotate(message_id, result)
    _respond(phone, ts, say, result, text)
    return sent


def _respond(phone, ts, say, result, original):
    decision = result.get("decision")
    template = result.get("template_id")

    if decision == "clarify":
        db.set_user(phone, pending_kind="clarify", pending_text=original)
        say(template or "clarify")
        return

    if decision == "escalate":
        say(template or "unsure")
        alert_agent(phone, ts, original, result=result)
        return

    say(template or "unsure")
    if result.get("label") in ("leaf_rust", "phoma"):
        scheduler.schedule(phone, "followup", clock.followup_due(ts))


def alert_agent(phone, ts, original, result=None, note=""):
    """The human safety net, and the eliminating criterion of the challenge.

    The agent needs enough to act without opening anything: who, what she
    actually wrote, the English of it, and what the model thought.
    """
    lines = [f"Coffee Advisor: {phone} needs a human."]
    if note:
        lines.append(note)
    lines.append(f"Said: {original}")
    if result:
        lines.append(f"English: {result.get('text_en') or original}")
        lines.append(f"Proposed: {result.get('label', '?')} (p={result.get('proba', 0):.2f})")
        if result.get("reason"):
            lines.append(f"Why: {result['reason']}")
    text = "\n".join(lines)

    db.event(phone, "agent_alert", text, ts)
    agent = os.environ.get("AGENT_PHONE", "")
    if agent:
        at_client.send_sms(agent, text)
    else:
        log.warning("AGENT_PHONE unset: no human was alerted for %s", phone)
