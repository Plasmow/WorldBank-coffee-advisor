# WorldBankAgriculture

Offline Small AI assistant helping smallholder coffee farmers diagnose crop problems and make better decisions, built for basic phones, low connectivity and local languages.

## Run it locally

`uv` and `--reload` are for a laptop. A host has neither: see **Deploy** for
the command servers actually run.

```bash
uv venv --python 3.12            # uv fetches 3.12 itself, no pyenv needed
uv pip install -r requirements.txt
uv run uvicorn app.main:app --reload
```

Then `http://localhost:8000/health` and `http://localhost:8000/docs`.

```bash
uv run pytest -q                 # the whole suite
uv run pytest -q -k followup     # one journey
uv run pytest -q tests/test_control_b.py   # the acceptance run
```

## Endpoints

| | |
|---|---|
| `POST /sms` | Africa's Talking webhook (form: `from`, `text`, `id`). Always answers `200` immediately; the analysis runs after the response. |
| `POST /ussd` | USSD menu (form: `phoneNumber`, `text`). 1 prices, 2 report a symptom, 3 talk to an agent. |
| `GET /health` | liveness |
| `GET /demo` | `web/index.html` when it exists, otherwise a backend-is-up page |
| `POST /api/demo/send` | `{phone, text}` — drives a demo number without touching the gateway |
| `GET /api/demo/state?phone=` | the whole transcript: messages with their analysis, plus agent alerts and follow-ups |
| `POST /api/demo/clock` | `{phone, day 0-6, slot morning\|day\|evening\|night}` — jump this visitor's clock |
| `POST /api/demo/reset` | `{phone}` |

## The AI chain

Off by default: `USE_REAL_AI=0` runs a keyword stand-in, so the backend boots
and the tests pass whatever state `ai/` is in. Flip it to `1` and `analyze()`
routes to `ai/analyze.py` instead — nothing else changes in `app/`.

Translation runs on **CTranslate2 int8**, not transformers. The same
NLLB-200-distilled-600M is ~2.4 GB in float32 and needs torch; converted once
it is ~600 MB, loads in seconds, and the server never installs torch at all.

```bash
uv pip install -r requirements-convert.txt   # torch, needed for this step only
uv run python scripts/convert_nllb_ct2.py    # ~2.4 GB in, ~600 MB out
```

The result lands in `model/nllb-ct2-int8/` and is gitignored — weights never
go in git. To get it onto Replit, push that folder to a Hugging Face repo and
set `NLLB_CT2_REPO`; the server downloads it once on first boot.

The model loads in the FastAPI lifespan, not on the first message: a farmer
must not wait out a cold start, and the gateway would have given up anyway.
`GET /health` reports resident memory and whether the model is really loaded.

If the chain is missing or throws, every message escalates to a human instead
of failing silently. A degraded demo beats a dead one.

## Environment

| variable | |
|---|---|
| `DB_PATH` | SQLite file, default `data/app.db`. Must be on persistent storage. |
| `DEMO_MODE` | `1` keeps every SMS off the gateway |
| `AT_USERNAME`, `AT_API_KEY`, `AT_SHORTCODE` | Africa's Talking sandbox |
| `AGENT_PHONE` | where the human safety net is paged |
| `FRONTEND_ORIGIN` | CORS origin of the Lovable page |
| `USE_REAL_AI` | `1` routes `analyze()` to `ai/` instead of the keyword stand-in |
| `NLLB_CT2_DIR` | converted model folder, default `model/nllb-ct2-int8` |
| `NLLB_CT2_REPO` | Hugging Face repo to pull the converted model from on first boot |
| `NLLB_COMPUTE_TYPE` | `int8` by default; `float32` to compare quality |
| `DEMO_LIVE_PHONE` | one real number the demo page may drive through the gateway |

Numbers starting `+256799` are demo numbers: their clock is simulated and
nothing addressed to them ever reaches Africa's Talking. The `/api/demo/*`
endpoints refuse every other number, because they are public once deployed:
otherwise anyone could read a farmer's conversation by guessing her number,
or spend our SMS credit.

`DEMO_LIVE_PHONE` opens exactly one exception. Set it and the demo page shows
a **Live SMS** switch that routes the conversation to that number through the
real gateway -- the last hop a demo number never takes. The number comes from
the environment, never from the request, so this opens one line rather than a
relay. Leave it unset in the public deployment, and remember `DEMO_MODE=1`
short-circuits every send (the page says so on the switch).

## Try the whole journey

```bash
P=+256799123456
curl -s -XPOST localhost:8000/api/demo/send -H 'content-type: application/json' \
     -d "{\"phone\":\"$P\",\"text\":\"my coffee leaves have orange powder\"}"
curl -s -XPOST localhost:8000/api/demo/clock -H 'content-type: application/json' \
     -d "{\"phone\":\"$P\",\"day\":3,\"slot\":\"evening\"}"      # the follow-up goes out
curl -s -XPOST localhost:8000/api/demo/send -H 'content-type: application/json' \
     -d "{\"phone\":\"$P\",\"text\":\"3\"}"                      # worse -> an agent is paged
```

## Deploy

**The command a host runs**, on Render or anywhere else — no `uv`, no
`--reload`, and bind the port the platform hands you or it will report no
open ports and fail the deploy:

```bash
pip install -r requirements.txt                          # build
uvicorn app.main:app --host 0.0.0.0 --port $PORT         # start
```

`render.yaml` already says this, but Render only reads it when the service is
created as a **Blueprint**. A manually created Web Service ignores the file
and uses the fields in the dashboard, so set both there.

Render's free tier is the default: `render.yaml` is committed, 512 MB is ten
times what the server needs with the AI off, and the URL survives closing the
laptop. It sleeps after 15 minutes idle and takes 30-60 s to wake, so ping
`/health` every 10 minutes during a demo window.

Install `requirements-ai.txt` instead (125 MB) only once `USE_REAL_AI=1` is
worth setting — that is, once `ai/analyze.py` actually exposes `analyze()`.
Until then the keyword stand-in gives better answers than a chain that fails
and escalates every message.

## Deploy on Replit

`.replit` is committed. Everything else is three steps.

### 1. Pick Reserved VM, not Autoscale

Reserved VM in the deployment UI, 2 GB of RAM at the very least — the
translation model alone peaks around 1.5 GB resident.

This is not a preference. Autoscale gives each instance a fresh filesystem and
scales to zero, so every cold start re-downloads the 621 MB model before it can
answer anything, and the gateway has long given up by then. Reserved VM keeps
its disk and never sleeps, which also keeps the background scheduler alive.
(`run_due()` still runs on every demo request anyway — belt and braces.)

### 2. Ship the model through the Hugging Face Hub

`model.bin` is 594 MB. GitHub refuses anything over 100 MB and weights do not
belong in git, so the Hub is the delivery channel. Once, from a laptop that has
already run `scripts/convert_nllb_ct2.py`:

```bash
uv run hf auth login                          # needs a write token
uv run hf upload <user>/nllb-200-distilled-600M-ct2-int8 model/nllb-ct2-int8 .
```

The three arguments are the repo, the local folder, and where it lands inside
the repo (`.` is the root). Keep the repo public, or add a read-only `HF_TOKEN`
secret — `huggingface_hub` picks it up from the environment on its own.

`ai.translate._ensure_model()` downloads it on first boot and only if
`model.bin` is not already on disk, so this costs nothing locally.

### 3. Secrets

Set these in the Replit Secrets pane, never in the repo:

```
AT_USERNAME, AT_API_KEY, AT_SHORTCODE    Africa's Talking sandbox
AGENT_PHONE                              where the human safety net is paged
FRONTEND_ORIGIN                          the Lovable page origin, for CORS
USE_REAL_AI=1                            once ai/analyze.py exposes analyze()
NLLB_CT2_REPO                            the Hub repo from step 2
DEMO_MODE=1                              keeps every SMS off the real gateway
```

### Check it

```bash
curl https://<app>.replit.app/health
# {"ok":true,"rss_mb":1422.3,"ai":{"enabled":true,"loaded":true,"seconds":2.6}}
```

`"loaded": false` comes with an `error` field saying why. The server boots
either way: a missing model is a degraded demo, a boot crash is no demo.

Then point the Africa's Talking sandbox at `https://<app>.replit.app/sms`
(and `/ussd` for the menu) and send a message from the simulator. That, not a
green test suite, is what proves the backend is live.
