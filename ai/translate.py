"""Luganda -> English, with our Opus-MT fine-tune converted to CTranslate2 int8.

Replaces NLLB-200-distilled-600M: that one was ~600 MB of weights, more than a
512 MB instance can hold, and it knew almost no farming Luganda. This model is
fine-tuned on SALT plus agricultural SMS and converted to int8, which brings it
to ~80 MB -- small enough to live on the same free instance as the web server.

One direction only. Opus-MT lg->en has no language tag and no reverse pass;
messages going out to Noor come from the `lg` side of data/templates.json, not
from a model.

Nothing heavy is imported at module level: a message written in English never
needs the model at all. The weights come from the Hub on first use, and a
failure to load degrades to passing the Luganda through untouched -- the
classifier then sees the glossary terms, and the server keeps answering.
"""

import logging
import os
import pathlib
import re
import time
from functools import lru_cache

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Adom4600/opus-lg-en-coffee-ct2"
DEFAULT_DIR = ROOT / "model" / "opus-lg-en-ct2"

_warned = False  # the "translator is down" line is worth saying once, not once per SMS


def model_dir():
    # Read at call time, not import time: tests and the host set this after import.
    return pathlib.Path(os.environ.get("TRANSLATOR_DIR") or DEFAULT_DIR)


def _ensure_model():
    path = model_dir()
    if (path / "model.bin").exists():
        return path

    # Weights never go in git, so a fresh host pulls them once from the Hub.
    repo = os.environ.get("TRANSLATOR_REPO") or DEFAULT_REPO
    from huggingface_hub import snapshot_download

    log.info("downloading %s into %s", repo, path)
    return pathlib.Path(snapshot_download(repo, local_dir=str(path)))


@lru_cache(maxsize=1)
def get_translator():
    path = _ensure_model()  # before the import: a slim install has no ctranslate2
    import ctranslate2

    compute_type = os.environ.get("TRANSLATOR_COMPUTE_TYPE") or "int8"
    log.info("loading CTranslate2 model from %s (%s)", path, compute_type)
    return ctranslate2.Translator(
        str(path), device="cpu", compute_type=compute_type,
        inter_threads=1, intra_threads=int(os.environ.get("TRANSLATOR_THREADS", "0")),
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    """The tokenizer saved next to the weights, so it cannot drift from them."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_ensure_model()))


def load():
    """Warm everything up at server startup. Returns seconds spent.

    The dummy sentence matters: the first translation is far slower than the
    rest, and Noor should not be the one paying for it. Raises if the model
    cannot be had -- app.analysis.preload() catches that and /health reports it.
    """
    started = time.monotonic()
    get_translator()
    get_tokenizer()
    _translate("Ebikoola bya kawa")
    elapsed = time.monotonic() - started
    log.info("translation model ready in %.1fs", elapsed)
    return elapsed


preload = load  # app/ calls load(), ai/ grew a preload(); same thing, both work.


def is_loaded():
    return get_translator.cache_info().currsize > 0


@lru_cache(maxsize=2048)
def _translate(text):
    text = (text or "").strip()
    if not text:
        return ""

    tokenizer = get_tokenizer()
    translator = get_translator()

    # CTranslate2 works on tokens, not ids. Opus-MT carries no language tag, so
    # there is no prefix to force and nothing to strip off the hypothesis.
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch(
        [source], beam_size=4, max_decoding_length=128,
        repetition_penalty=1.2, no_repeat_ngram_size=3,
    )
    hypothesis = results[0].hypotheses[0]
    return tokenizer.decode(
        tokenizer.convert_tokens_to_ids(hypothesis), skip_special_tokens=True
    )


def lug_to_en(text):
    """Never raises. A missing or broken model returns the Luganda untouched:
    the glossary below still gives the classifier something to work with, and a
    farmer gets a clarifying question instead of silence."""
    global _warned
    try:
        return _translate(text)
    except Exception:
        if not _warned:
            log.exception("translator unavailable; Luganda will pass through untranslated")
            _warned = True
        return text


# The model knows farming Luganda far better than NLLB did, but the words the
# classifier keys on are worth guaranteeing: they are matched by stem and
# appended in English when the translation drops them.
GLOSSARY = {
    "mmwanyi": "coffee", "kaawa": "coffee", "kawa": "coffee",
    "bikoola": "leaves", "kikoola": "leaf",
    "buwunga": "powder", "nfuufu": "dust",
    "kacungwa": "orange", "kyenvu": "yellow", "zirugavu": "black",
    "ddugavu": "black", "kitaka": "brown", "kiragala": "green",
    "mabala": "spots", "bbala": "spot", "bubonero": "marks",
    "wansi": "underneath", "nsonda": "tips", "mbiriizi": "edges",
    "kala": "drying", "gwa": "falling", "bigwa": "falling",
    "bibala": "berries", "matabi": "branches", "kikolo": "stem",
    "mirandira": "roots", "biwuka": "insects", "butuli": "holes",
    "mpewo": "cold", "nkuba": "rain", "bulungi": "fine",
    "masamasa": "shiny",
}


def glossary_terms(text):
    """English for the farming words found in a Luganda message, in order."""
    terms = []
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        for stem, en in GLOSSARY.items():
            # stems match inside a word (noun-class prefixes vary: e-bi-koola,
            # zi-kala), except the very short ones, which must match a suffix
            if (stem in word if len(stem) > 3 else word.endswith(stem)) and en not in terms:
                terms.append(en)
                break
    return terms


def lug_to_en_with_terms(text):
    """The translation, followed by the glossary terms it may have lost."""
    english = lug_to_en(text)
    missing = [t for t in glossary_terms(text) if t not in english.lower()]
    return f"{english} ({', '.join(missing)})" if missing else english


def en_to_lug(text):
    raise NotImplementedError(
        "The Opus-MT model only translates Luganda to English. Messages going "
        "out to Noor must come from the 'lg' side of data/templates.json, "
        "never from a model."
    )
