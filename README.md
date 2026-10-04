# Coffee Advisor

**An SMS crop-disease advisor for smallholder coffee farmers on basic phones, in Luganda and English.**

Built for the Hack-Nation × World Bank challenge *Small AI for Development* (Agriculture track).

Noor grows arabica on Mount Elgon, Uganda. She has a basic phone, no data plan,
and speaks Luganda. She texts what she sees on her plants; Coffee Advisor
answers by SMS with vetted advice, asks one question when the message is
unclear, and hands her over to a human extension agent whenever it is not sure.

- **Small AI only**: a 600M-parameter translator quantized to int8, a 33M-parameter
  embedder with a calibrated linear head, and an optional 3B local LLM as a
  second opinion. Everything runs on a CPU, no cloud LLM ever talks to the farmer.
- **No generated text reaches the farmer**: the models only pick an id; every
  reply is a fixed, reviewed template from `data/templates.json`.
- **Human in the loop**: unclear twice, out of scope, or "3 = worse" at the
  follow-up three days later, and the agent is paged by SMS.
- **Works on any phone**: SMS and USSD through Africa's Talking, plus `PRICE`
  (official UCDA prices, no AI), `HELP`, `STOP` / `START`.

## Repository

<<<<<<< HEAD
```bash
uv venv --python 3.12            # uv fetches 3.12 itself, no pyenv needed
uv pip install -r requirements-ai.txt
uv run python scripts/fetch_models.py      # translator (~80 MB) + e5-small
ollama pull qwen2.5:3b                     # optional: the LLM second opinion
cp .env.example .env                       # then fill in AT_API_KEY and AGENT_PHONE
uv run uvicorn app.main:app --reload
=======
>>>>>>> 67abd3e46b7a98681f5585ccb36c4b7132d48ecb
```
app/      FastAPI backend: SMS/USSD webhooks, router, SQLite, scheduler, demo API
ai/       the AI chain: language id, translation, classifier, LLM second opinion
data/     reply templates (en, lg), coffee prices, synthetic training data, embeddings
model/    the trained classifier head (numpy) and its labels/threshold
scripts/  data generation, training, evaluation, model download and conversion
web/      the demo page (phone simulator + agent view), served at /demo
docs/     datasheet of the data and the model
deploy/   Render blueprint notes, systemd unit and Caddyfile for a VM
tests/    pytest suite (models stubbed, runs in seconds)
```

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
| `GOOGLE_TRANSLATE_API_KEY` | optional: English subtitles of the Luganda conversation on the demo page. Empty = Google's free endpoint |

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

## Install and run locally

You need Python 3.10+ (3.12 recommended), about 4 GB of free disk and 2 GB of
RAM for the full chain. The commands use [`uv`](https://docs.astral.sh/uv/);
plain `python -m venv` + `pip` work the same way.

**1. Dependencies**

```bash
git clone https://github.com/Plasmow/WorldBankAgriculture.git
cd WorldBankAgriculture
uv venv --python 3.12
uv pip install -r requirements-ai.txt     # full chain: CTranslate2, transformers, CPU torch, ollama
# or: uv pip install -r requirements.txt  # web server only, keyword stand-in (USE_REAL_AI=0)
```

**2. Models.** Weights are never in git. Pick one of the two ways to get the
translator, quantized to int8:

```bash
# a) download a ready-made int8 conversion (~620 MB) + tokenizer + e5-small-v2
uv run python scripts/fetch_models.py

# b) or quantize it yourself from facebook/nllb-200-distilled-600M
#    (downloads ~2.4 GB, writes ~600 MB into model/nllb-ct2-int8/)
uv pip install -r requirements-convert.txt
uv run python scripts/convert_nllb_ct2.py   # ct2-transformers-converter --quantization int8
uv run python scripts/fetch_models.py       # still needed for the tokenizer and e5-small-v2
```

The classifier head (`model/classifier.npz`) is already in the repo. To
rebuild it from the data (optional; needs `scikit-learn scipy joblib
sentence-transformers`):

```bash
uv pip install scikit-learn scipy joblib sentence-transformers
uv run python scripts/extract_embeddings.py
uv run python scripts/train_classifier.py
uv run python scripts/evaluate.py
uv run python scripts/export_classifier.py   # joblib -> numpy, what the server reads
```

The LLM second opinion is optional. Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:3b      # ~1.9 GB, already 4-bit quantized (Q4_K_M)
ollama serve                # if it is not already running as a service
```

Without Ollama the calibrated classifier answers alone and uncertain cases
still go to the agent.

**3. `.env`.** Copy `cp .env.example .env` and fill it in. For a fully local
run, with no SMS ever leaving the machine:

```dotenv
# Africa's Talking: any value works while DEMO_MODE=1
AT_USERNAME=sandbox
AT_API_KEY=your-sandbox-api-key
AT_SHORTCODE=
AGENT_PHONE=+256700000099        # the extension agent paged by the safety net

DEMO_MODE=1                      # 1 = never call Africa's Talking
USE_REAL_AI=1                    # 0 = keyword stand-in, no model needed
USE_LLM=1                        # 0 = skip the Ollama second opinion
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
NLLB_CT2_REPO=JustFrederik/nllb-200-distilled-600M-ct2-int8

GOOGLE_TRANSLATE_API_KEY=        # optional, English subtitles for judges on /demo
FRONTEND_ORIGIN=*
DB_PATH=data/app.db
```

To send real SMS through the Africa's Talking sandbox, set `DEMO_MODE=0`,
your sandbox `AT_API_KEY`, and point the sandbox's incoming-SMS callback at
`<public-url>/sms` (e.g. through `cloudflared tunnel --url http://localhost:8000`).

**4. Run**

```bash
uv run uvicorn app.main:app --reload
```

- `http://localhost:8000/demo`: the phone simulator, the agent's view, and a
  clock to jump to the day-3 follow-up
- `http://localhost:8000/health`: shows `"loaded": true` once the models are in (~30 s)
- `http://localhost:8000/docs`: the API

**5. Tests**

```bash
uv run pytest -q                                          # models stubbed, seconds
RUN_MODEL_TESTS=1 uv run pytest -q tests/test_models.py   # the real models (~1 min)
uv run python scripts/report_chain.py                     # quality report on Luganda and English cases
```
