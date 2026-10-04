# WorldBankAgriculture

Offline Small AI assistant helping smallholder coffee farmers diagnose crop problems and make better decisions, built for basic phones, low connectivity and local languages.

## Run it

```bash
uv venv --python 3.12            # Replit runs 3.12; uv fetches it, no pyenv needed
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

Numbers starting `+256799` are demo numbers: their clock is simulated and
nothing addressed to them ever reaches Africa's Talking.

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

Replit, one worker, one instance. `.replit` is committed; set the environment
variables as Replit secrets, never in the repo. The background scheduler loop
dies when the instance sleeps, so `run_due()` also runs on every demo request.
