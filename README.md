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

## Environment

| variable | |
|---|---|
| `DB_PATH` | SQLite file, default `data/app.db`. Must be on persistent storage. |
| `DEMO_MODE` | `1` keeps every SMS off the gateway |
| `AT_USERNAME`, `AT_API_KEY`, `AT_SHORTCODE` | Africa's Talking sandbox |
| `AGENT_PHONE` | where the human safety net is paged |
| `FRONTEND_ORIGIN` | CORS origin of the Lovable page |
| `USE_REAL_AI` | `1` routes `analyze()` to `ai/` instead of the keyword stand-in |

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
