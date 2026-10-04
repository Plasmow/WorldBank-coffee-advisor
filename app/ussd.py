"""USSD menu, for the farmer who would rather navigate than type a sentence.

Africa's Talking replays the whole session in `text`, segments joined by '*',
so nothing has to be remembered between screens. A reply starts with CON to
keep the session open or END to hang up, and AT truncates past ~182 chars.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from app import clock, content, db, router

log = logging.getLogger(__name__)
api = APIRouter()
SCREEN = 182


@api.post("/ussd")
async def ussd(request: Request):
    data = {}
    try:
        data = dict(await request.form())
    except Exception:
        pass
    if not data:
        try:
            data = await request.json()
        except Exception:
            data = dict(request.query_params)

    phone = db.norm_phone(str(data.get("phoneNumber") or ""))
    text = str(data.get("text") or "").strip()
    if not phone:
        return PlainTextResponse("END Sorry, we could not identify your number.")

    try:
        body = _screen(phone, text)
    except Exception:
        log.exception("ussd failed for %s", phone)
        body = "END Sorry, something went wrong. Please try again."
    return PlainTextResponse(body[:SCREEN])


def _screen(phone, text):
    parts = text.split("*") if text else []
    choice = parts[0] if parts else ""

    if choice == "1":
        return "END " + content.prices_text()

    if choice == "2":
        symptom = parts[1].strip() if len(parts) > 1 else ""
        if not symptom:
            return "CON " + content.t("ussd_ask_symptom")
        # Same path as an SMS: one diagnosis, one escalation rule, one follow-up.
        replies = router.handle_incoming(phone, symptom)
        return "END " + (replies[-1] if replies else content.t("unsure"))

    if choice == "3":
        db.user(phone)
        router.alert_agent(
            phone, clock.now(phone), "asked for an agent", note="requested from the USSD menu"
        )
        return "END " + content.t("agent_requested")

    return "CON " + content.t("ussd_menu")
