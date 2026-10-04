"""English subtitles for the demo page, from Google Translate.

For the judges only: Noor never sees this, and nothing here feeds the AI chain
(that runs on our own Opus-MT model, on our own server). With GOOGLE_TRANSLATE_API_KEY set this
calls the Cloud Translation API v2; without it, the free endpoint Google's own
web widgets use, which needs no key and is fine for a demo.
"""

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

CLOUD_URL = "https://translation.googleapis.com/language/translate/v2"
FREE_URL = "https://translate.googleapis.com/translate_a/single"
MAX_CHARS = 1000
COOLDOWN_S = 60  # after a 429 without Retry-After
ERROR_COOLDOWN_S = 15

# No call to Google before this instant (time.monotonic()). The free endpoint
# answers 429 to anyone who keeps asking; retrying on every poll made it worse.
_blocked_until = 0.0

# text -> (english, detected source language). Demo texts repeat a lot (the
# templates), and the page asks again on every new message.
_cache = {}


def to_english(texts):
    """-> [(english or None, source lang or None)], same order as texts.

    None means Google could not be reached; that text is retried next time.
    """
    out = []
    for text in texts:
        text = (text or "")[:MAX_CHARS]
        if text not in _cache:
            result = _translate(text)
            if result[0] is None:
                out.append(result)
                continue
            _cache[text] = result
        out.append(_cache[text])
    return out


def _translate(text):
    if not text.strip() or not any(c.isalpha() for c in text):
        return text, None  # "3", "1": nothing to translate
    from ai.lang import detect_lang  # pure Python, no model

    if detect_lang(text, default="lg") == "en":
        # Already English: Google would only damage it (grade names like
        # "Kiboko" come back as "Kick").
        return text, "en"
    global _blocked_until
    if time.monotonic() < _blocked_until:
        return None, None  # cooling down: the caller falls back, no request made
    try:
        key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip()
        return _cloud(text, key) if key else _free(text)
    except httpx.HTTPStatusError as e:
        wait = COOLDOWN_S if e.response.status_code == 429 else ERROR_COOLDOWN_S
        try:
            wait = max(wait, int(e.response.headers.get("Retry-After", 0)))
        except ValueError:
            pass
        _blocked_until = time.monotonic() + wait
        log.warning("google translate %s, pausing it for %ss", e.response.status_code, wait)
        return None, None
    except Exception as e:
        _blocked_until = time.monotonic() + ERROR_COOLDOWN_S
        log.warning("google translate failed (%s), pausing it for %ss", type(e).__name__, ERROR_COOLDOWN_S)
        return None, None


def _cloud(text, key):
    r = httpx.post(CLOUD_URL, params={"key": key},
                   data={"q": text, "target": "en", "format": "text"}, timeout=5)
    r.raise_for_status()
    t = r.json()["data"]["translations"][0]
    return t["translatedText"], t.get("detectedSourceLanguage")


def _free(text):
    r = httpx.get(FREE_URL, params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
                  timeout=5)
    r.raise_for_status()
    data = r.json()
    english = "".join(part[0] for part in data[0] if part and part[0])
    return english, data[2] if len(data) > 2 else None
