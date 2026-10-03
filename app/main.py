import asyncio
import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from app import db, scheduler
from app.demo_routes import api
from app.router import handle_incoming
from app.ussd import api as ussd_api

log = logging.getLogger(__name__)
LOOP_SECONDS = 60


async def _background():
    while True:
        await asyncio.sleep(LOOP_SECONDS)
        try:
            scheduler.run_due()
        except Exception:
            log.exception("scheduler loop")


@asynccontextmanager
async def lifespan(app):
    db.init()
    task = asyncio.create_task(_background())
    yield
    task.cancel()


app = FastAPI(title="Coffee Advisor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("FRONTEND_ORIGIN", "*").split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api)
app.include_router(ussd_api)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/sms")
async def sms(request: Request, background: BackgroundTasks):
    """Africa's Talking webhook.

    Parsed by hand rather than with Form(...): a missing field must not become
    a 422. AT retries anything that is not a 2xx, so this always answers 200,
    and it answers before the analysis runs -- translation and classification
    are far slower than the gateway's patience.
    """
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

    frm = str(data.get("from") or data.get("msisdn") or "")
    text = str(data.get("text") or "")
    at_id = str(data.get("id") or "")

    if not frm.strip() or not text.strip():
        return PlainTextResponse("ok")
    if not db.seen_once(at_id):
        return PlainTextResponse("ok")  # AT replayed a message we already handled

    background.add_task(_handle, frm, text)
    return PlainTextResponse("ok")


def _handle(frm, text):
    try:
        handle_incoming(frm, text)
    except Exception:
        log.exception("handle_incoming failed for %s", frm)


@app.get("/demo", include_in_schema=False)
def demo_page():
    """Fallback demo page. The live one is on Lovable; web/index.html is the
    spare tyre, and P3 owns it."""
    page = pathlib.Path(__file__).resolve().parent.parent / "web" / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text())
    return HTMLResponse(
        "<p>Demo page not built yet. The backend is up: "
        "<a href='/docs'>/docs</a>, <a href='/health'>/health</a>.</p>"
    )
