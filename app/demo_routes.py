"""Demo surface for the Lovable page. Each visitor gets a +256799... number
and their own simulated clock; nothing here ever reaches Africa's Talking."""

from fastapi import APIRouter
from pydantic import BaseModel

from app import clock, db, router, scheduler

api = APIRouter(prefix="/api/demo")


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
    phone = db.norm_phone(body.phone)
    scheduler.run_due(phone)  # the instance may have slept through the due time
    replies = router.handle_incoming(phone, body.text)
    return {"replies": replies, **state(phone)}


@api.get("/state")
def state(phone: str):
    phone = db.norm_phone(phone)
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


@api.post("/clock")
def set_clock(body: SetClock):
    phone = db.norm_phone(body.phone)
    if body.slot not in clock.SLOT_HOUR:
        return {"error": f"slot must be one of {sorted(clock.SLOT_HOUR)}"}
    db.user(phone)
    db.set_user(phone, day=max(0, min(6, body.day)), slot=body.slot)
    scheduler.run_due(phone)  # the jump may have made a follow-up due
    return state(phone)


@api.post("/reset")
def reset(body: Phone):
    phone = db.norm_phone(body.phone)
    db.reset(phone)
    return {"ok": True}
