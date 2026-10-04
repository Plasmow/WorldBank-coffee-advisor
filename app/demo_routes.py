"""Demo surface for the Lovable page. Each visitor gets a +256799... number
and their own simulated clock; nothing here ever reaches Africa's Talking."""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import clock, content, db, gtranslate, router, scheduler

api = APIRouter(prefix="/api/demo")


def live_phone():
    """The one real number the page may drive, named by the operator.

    Set DEMO_LIVE_PHONE and the page can push a message all the way through
    Africa's Talking -- the last hop a demo number never takes. The number
    comes from the environment and never from the request, so this opens one
    line, not a relay to any number on earth.
    """
    raw = os.environ.get("DEMO_LIVE_PHONE", "").strip()
    return db.norm_phone(raw) if raw else None


def demo_phone(raw):
    """Normalise, and refuse anything the operator has not opened.

    These routes are public on Replit. Without this, anyone could read a real
    farmer's whole conversation by guessing her number, or make the server
    send SMS on our Africa's Talking credit.
    """
    phone = db.norm_phone(raw)
    if clock.is_demo(phone) or (phone and phone == live_phone()):
        return phone
    raise HTTPException(400, f"demo numbers only (must start with {clock.DEMO_PREFIX})")


@api.get("/config")
def config():
    """What the page is allowed to do here, so it can show the right controls."""
    return {
        "live_phone": live_phone(),
        "demo_mode": os.environ.get("DEMO_MODE") == "1",
        "demo_prefix": clock.DEMO_PREFIX,
    }


class Send(BaseModel):
    phone: str
    text: str


class SetClock(BaseModel):
    phone: str
    day: int = 0
    slot: str = "morning"


class Phone(BaseModel):
    phone: str


@api.post("/send")
def send(body: Send):
    phone = demo_phone(body.phone)
    scheduler.run_due(phone)  # the instance may have slept through the due time
    replies = router.handle_incoming(phone, body.text)
    return {"replies": replies, **state(phone)}


@api.get("/state")
def state(phone: str):
    phone = demo_phone(phone)
    scheduler.run_due(phone)
    return {
        "now": clock.iso(clock.now(phone)),
        "messages": [
            {
                "id": m["id"],
                "direction": m["direction"],
                "text": m["text"],
                "ts": clock.iso(m["ts"]),
                "text_en": m["text_en"],
                "label": m["label"],
                "proba": m["proba"],
                "decision": m["decision"],
            }
            for m in db.messages(phone)
        ],
        "events": [
            {"id": e["id"], "kind": e["kind"], "text": e["text"], "ts": clock.iso(e["ts"])}
            for e in db.events(phone)
        ],
    }


@api.get("/english")
def english(phone: str):
    """The conversation in English, for the judges.

    - our replies: the exact English of the template they came from;
    - Noor's messages: Google Translate, or, when Google is unavailable,
      the NLLB translation the AI chain already made of them.

    Takes a phone, not free text: the route translates only what is already
    in a demo conversation, so it cannot be used as a free translation proxy.
    """
    phone = demo_phone(phone)
    rows = db.messages(phone)
    out = []
    for m in rows:
        en = content.english_of(m["text"]) if m["direction"] == "out" else None
        if en is not None:
            src = "en" if en == m["text"] else "lg"
            out.append({"id": m["id"], "direction": m["direction"], "text": m["text"],
                        "en": en, "src": src, "via": "template"})
            continue
        (en, src), = gtranslate.to_english([m["text"]])
        via = "google"
        if en is None and m["text_en"]:
            en, src, via = m["text_en"], "lg", "nllb"
        out.append({"id": m["id"], "direction": m["direction"], "text": m["text"],
                    "en": en, "src": src, "via": via if en is not None else None})
    return {"messages": out}


@api.post("/clock")
def set_clock(body: SetClock):
    phone = demo_phone(body.phone)
    if body.slot not in clock.SLOT_HOUR:
        return {"error": f"slot must be one of {sorted(clock.SLOT_HOUR)}"}
    db.user(phone)
    db.set_user(phone, day=max(0, min(6, body.day)), slot=body.slot)
    scheduler.run_due(phone)  # the jump may have made a follow-up due
    return state(phone)


@api.post("/reset")
def reset(body: Phone):
    phone = demo_phone(body.phone)
    db.reset(phone)
    return {"ok": True}
