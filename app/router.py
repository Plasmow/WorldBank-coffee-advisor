"""One entry point for every inbound message, whatever the channel."""

import logging
import os
import re
import threading
import unicodedata
from collections import defaultdict

from app import at_client, clock, content, db, scheduler
from ai.lang import detect_lang  # pure Python, no model
from app.analysis import analyze

log = logging.getLogger(__name__)

MAX_IN = 1000  # a gateway can hand us anything; the classifier does not need more

# English, French (the spec) and Luganda spellings. A few extra strings beat
# getting this wrong on stage.
KEYWORDS = {
    "PRICE": ("price", "prices", "prix", "bbeeyi", "beeyi"),
    "HELP": ("help", "aide", "yamba", "nnyamba"),
    "AGENT": ("agent", "omulimisa"),
    "STOP": ("stop", "komya"),
    "START": ("start", "tandika"),
}
# Luganda keyword -> reply in Luganda; anything else keeps the farmer's language.
LG_KEYWORDS = {"bbeeyi", "beeyi", "yamba", "nnyamba", "omulimisa", "komya", "tandika"}

# A question about prices in free text gets prices.json, never the classifier.
# Checked on what she wrote and again on its English translation.
PRICE_QUESTION = re.compile(
    r"\b(price|prices|prix|cost|costs|sell|selling|market rate|how much|pay for"
    r"|bbeeyi|beeyi|ssente mmeka|emiwendo|bagula|tunda)\b",
    re.IGNORECASE,
)
FOLLOWUP_ANSWERS = {"1": "better", "2": "same", "3": "worse"}


def _norm(text):
    w = unicodedata.normalize("NFKD", text.strip().lower())
    return w.encode("ascii", "ignore").decode().strip(" .!?")


def keyword(text):
    """Exact match on the whole message, so "help my coffee is dying" is a
    symptom report and not the HELP menu."""
    w = _norm(text)
    for name, words in KEYWORDS.items():
        if w in words:
            return name
    return None


def is_price_question(text):
    return bool(PRICE_QUESTION.search(text or ""))


# One conversation at a time per phone. Two SMS sent back to back reach the
# webhook together; processed in parallel, both would see "not welcomed yet"
# and both would answer a clarifying question that was asked only once.
_phone_locks = defaultdict(threading.Lock)


def handle_incoming(phone, text):
    """Returns the messages actually sent to this phone."""
    phone = db.norm_phone(phone)
    text = (text or "").strip()[:MAX_IN]
    if not phone or not text:
        return []
    with _phone_locks[phone]:
        return _handle(phone, text)


def _handle(phone, text):
    ts = clock.now(phone)
    row, _ = db.user(phone)
    message_id = db.record(phone, "in", text, ts)

    kw = keyword(text)
    # Reply in the language she writes in. A keyword says little about that,
    # except a Luganda one; "1"/"2"/"3" says nothing and keeps the last one.
    lang = row["lang"]
    if kw:
        lang = "lg" if _norm(text) in LG_KEYWORDS else lang
    elif not (row["pending_kind"] == "followup" and text.strip() in FOLLOWUP_ANSWERS):
        lang = detect_lang(text, default=lang)
    if lang != row["lang"]:
        db.set_user(phone, lang=lang)
        row = db.get_user(phone)

    sent = []

    def say(key_or_text, literal=False):
        msg = key_or_text if literal else content.t(key_or_text, row["lang"])
        if msg and db.send(phone, msg, ts):
            sent.append(msg)

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

    if kw == "PRICE" or (kw is None and is_price_question(text)):
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

    # The translation can reveal a price question the raw text hid
    # ("Emmwanyi zigula ssente ki?" -> "How much does coffee sell for?").
    if result.get("label") == "other" and is_price_question(result.get("text_en")):
        say(content.prices_text(db.get_user(phone)["lang"]), literal=True)
        return

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
