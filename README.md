# WorldBankAgriculture

Offline Small AI assistant helping smallholder coffee farmers diagnose crop problems and make better decisions, built for basic phones, low connectivity and local languages.

## Run it locally

`uv` and `--reload` are for a laptop. A host has neither: see **Deploy** for
the command servers actually run.

```bash
uv venv --python 3.12            # uv fetches 3.12 itself, no pyenv needed
uv pip install -r requirements-ai.txt
uv run python scripts/fetch_models.py      # translator (~80 MB) + e5-small
ollama pull qwen2.5:3b                     # optional: the LLM second opinion
cp .env.example .env                       # then fill in AT_API_KEY and AGENT_PHONE
uv run uvicorn app.main:app --reload
```

`requirements.txt` alone is enough for the keyword stand-in (`USE_REAL_AI=0`).

Then `http://localhost:8000/health` and `http://localhost:8000/docs`.

```bash
uv run pytest -q                 # the whole suite
uv run pytest -q -k followup     # one journey
uv run pytest -q tests/test_control_b.py   # the acceptance run
RUN_MODEL_TESTS=1 uv run pytest -q tests/test_models.py   # the real models (~1 min)
```

The suite stubs every model, so it runs in seconds anywhere. Run the model
tests before any deploy that touches `ai/` or `model/`.

## Endpoints

| | |
|---|---|
| `POST /sms` | Africa's Talking webhook (form: `from`, `text`, `id`). Always answers `200` immediately; the analysis runs after the response. |
| `POST /ussd` | USSD menu (form: `phoneNumber`, `text`). 1 prices, 2 report a symptom, 3 talk to an agent. |
| `GET /health` | liveness, peak memory, and whether the models are loaded |
| `GET /demo` | `web/index.html` when it exists, otherwise a backend-is-up page |
| `POST /api/demo/send` | `{phone, text}` — drives a demo number without touching the gateway |
| `GET /api/demo/state?phone=` | the whole transcript: messages with their analysis, plus agent alerts and follow-ups |
| `POST /api/demo/clock` | `{phone, day 0-6, slot morning\|day\|evening\|night}` — jump this visitor's clock |
| `POST /api/demo/reset` | `{phone}` |

## The AI chain

```
SMS -> keyword? (PRICE/HELP/AGENT/STOP/START, 1/2/3) -> direct reply, no AI
    -> price question in free text?                   -> prices.json, no AI
    -> Luganda? -> Opus-MT lg->en fine-tune (CT2 int8) + glossary     -> English
    -> e5-small-v2 + calibrated logistic regression   -> label, proba
    -> qwen2.5:3b via Ollama (JSON schema, temp 0)    -> label, which question to ask
    -> answer   if proba >= 0.8 and both agree        -> advice template, follow-up at D+3
       clarify  otherwise, once                       -> the question the LLM picked
       escalate still unsure, or "other"              -> "not sure" + SMS to the agent
```

- Every word sent to Noor comes from `data/templates.json` (English and
  Luganda, `lg_verified: false` until a native speaker checks them). The models
  only choose a template id. Replies follow the language of her last message.
- `other` is never answered. A price question, by keyword or in free text
  ("Emmwanyi zigula ssente mmeka leero?"), gets `prices.json` in a fixed
  wrapper and never reaches a model.
- The translator is our own Opus-MT fine-tuned on SALT plus agricultural
  SMS, converted to CTranslate2 int8: 81 MB on disk, 2.2 s to load, 105 ms a
  message, chrF 65.0 on `data/synthetic/domain_pairs.jsonl`
  (`python scripts/eval_translate.py`). It replaced NLLB-200-600M, which was
  600 MB and rendered "my coffee is fine" as "my oil is good".
- It only goes Luganda to English. Everything sent to Noor is written by hand
  in `data/templates.json`; `en_to_lug` raises rather than invent a reply.
- `ai/translate.py` appends the English of the farming words it recognises, so
  the classifier still sees "powder, orange, underneath" even on a bad line.
  A model that cannot be loaded returns the Luganda untouched: the glossary
  carries the message and Noor gets a clarifying question, never silence.
- The classifier head is read from `model/classifier.npz` (numpy, no sklearn
  at runtime). Re-export it after retraining: `python scripts/export_classifier.py`.
- No Ollama reachable: the calibrated classifier answers alone (plan B of the
  roadmap) and the generic question is used. Anything that throws sends the
  farmer to a human.

`USE_REAL_AI=0` swaps the whole chain for a keyword stand-in, so the backend
boots and the tests pass without any model.

## Environment

See `.env.example`.

| variable | |
|---|---|
| `AT_USERNAME`, `AT_API_KEY`, `AT_SHORTCODE` | Africa's Talking. `sandbox` uses the sandbox API; leave the shortcode empty there |
| `AGENT_PHONE` | where the human safety net is paged |
| `DEMO_MODE` | `1` keeps every SMS off the gateway. `0` on the deployed server |
| `USE_REAL_AI` | `1` = the chain above, `0` = keyword stand-in |
| `USE_LLM`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` | the second opinion |
| `TRANSLATOR_DIR`, `TRANSLATOR_REPO` | translator folder / Hub repo it is fetched from (default `Adom4600/opus-lg-en-coffee-ct2`) |
| `CLASSIFIER_THRESHOLD` | overrides `model/labels.json` |
| `DB_PATH` | SQLite file, default `data/app.db` |
| `FRONTEND_ORIGIN` | CORS origin of the Lovable page |
| `DEMO_LIVE_PHONE` | one real number the demo page may drive through the gateway |

Numbers starting `+256799` are demo numbers: their clock is simulated and
nothing addressed to them ever reaches Africa's Talking. The `/api/demo/*`
endpoints refuse every other number, because they are public once deployed.
`DEMO_LIVE_PHONE` opens exactly one exception, taken from the environment,
never from the request.

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

## Deploy on Render

`render.yaml` is a Blueprint: **New > Blueprint**, pick the repo, then fill in
the secrets it asks for (`AT_API_KEY`, `AGENT_PHONE`, optionally
`OLLAMA_HOST`, `DEMO_LIVE_PHONE`).

- **Plan: Standard (2 GB).** The translator alone is small enough for the
  free plan, but e5-small and its torch are not.
- **Build** installs `requirements-ai.txt` (CPU-only torch) and runs
  `scripts/fetch_models.py`, which downloads the translator (~80 MB) and
  e5-small into the project folder. A boot then only reads from disk (~30 s), in a
  background thread: `/health` answers at once and shows `"loading": true`
  until the models are in.
- **The LLM** needs an Ollama server Render can reach: set `OLLAMA_HOST` to,
  e.g., a laptop running `ollama serve` behind `cloudflared tunnel --url
  http://localhost:11434`. Without it the classifier answers alone.

Check it, then point the sandbox at it:

```bash
curl https://<app>.onrender.com/health
# {"ok":true,"rss_mb":1371.6,"ai":{"loaded":true,"loading":false,"seconds":26.8,"enabled":true}}
```

In the Africa's Talking sandbox: **SMS > SMS Callback URLs > Incoming
messages** = `https://<app>.onrender.com/sms`, and the USSD callback =
`https://<app>.onrender.com/ussd`. Then send a message from the simulator.
