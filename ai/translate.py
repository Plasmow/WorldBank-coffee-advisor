"""Luganda <-> English, with NLLB-200-distilled-600M converted to CTranslate2 int8.

Why not transformers at runtime. In float32 the model is ~2.4 GB and needs
torch for generation; converted once to CTranslate2 int8 it is ~600 MB, loads
in seconds and runs several times faster on CPU. Only the tokenizer comes from
transformers.

Nothing heavy is imported at module level: a message written in English never
needs the model at all. Convert once with scripts/convert_nllb_ct2.py, or set
NLLB_CT2_REPO to a converted copy on the Hugging Face Hub, then call load()
from the server's lifespan.
"""

import logging
import os
import pathlib
import time
from functools import lru_cache

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER_NAME = os.environ.get("NLLB_TOKENIZER") or "facebook/nllb-200-distilled-600M"
# First match wins: where scripts/convert_nllb_ct2.py writes, then the older local copy.
DEFAULT_DIRS = (ROOT / "model" / "nllb-ct2-int8", ROOT / "ai" / "nllb-600m-ct2")
LUG, ENG = "lug_Latn", "eng_Latn"


def model_dir():
    # Read at call time, not import time: tests and the host set this after import.
    env = os.environ.get("NLLB_CT2_DIR")
    if env:
        return pathlib.Path(env)
    for path in DEFAULT_DIRS:
        if (path / "model.bin").exists():
            return path
    return DEFAULT_DIRS[0]


def _ensure_model():
    path = model_dir()
    if (path / "model.bin").exists():
        return path

    # Weights never go in git, so a fresh host pulls them once from the Hub.
    repo = os.environ.get("NLLB_CT2_REPO") or ""
    if repo:
        from huggingface_hub import snapshot_download

        log.info("downloading %s into %s", repo, path)
        return pathlib.Path(snapshot_download(repo, local_dir=str(path)))

    raise FileNotFoundError(
        f"No CTranslate2 model in {path}. Run scripts/convert_nllb_ct2.py once, "
        f"or set NLLB_CT2_REPO to a converted model on the Hugging Face Hub."
    )


@lru_cache(maxsize=1)
def get_translator():
    path = _ensure_model()  # before the import: the slim install has no ctranslate2
    import ctranslate2

    compute_type = os.environ.get("NLLB_COMPUTE_TYPE") or "int8"
    log.info("loading CTranslate2 model from %s (%s)", path, compute_type)
    return ctranslate2.Translator(
        str(path), device="cpu", compute_type=compute_type,
        inter_threads=1, intra_threads=int(os.environ.get("NLLB_THREADS", "0")),
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    """One tokenizer for both directions: only the source language tag changes.

    Always the original model's (17 MB, cached by scripts/fetch_models.py):
    copies shipped next to converted weights come in several formats.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def load():
    """Warm everything up at server startup. Returns seconds spent.

    The dummy sentence matters: the first translation is far slower than the
    rest, and Noor should not be the one paying for it.
    """
    started = time.monotonic()
    get_translator()
    get_tokenizer()
    lug_to_en("Ebikoola bya kawa")
    elapsed = time.monotonic() - started
    log.info("translation model ready in %.1fs", elapsed)
    return elapsed


preload = load  # app/ calls load(), ai/ grew a preload(); same thing, both work.


def is_loaded():
    return get_translator.cache_info().currsize > 0


@lru_cache(maxsize=2048)
def _translate(text, src_lang, tgt_lang):
    text = (text or "").strip()
    if not text:
        return ""

    tokenizer = get_tokenizer()
    tokenizer.src_lang = src_lang
    translator = get_translator()

    # CTranslate2 works on tokens, not ids. The source carries its language tag
    # from the tokenizer; the target one is forced as a prefix.
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch(
        [source], target_prefix=[[tgt_lang]], beam_size=4,
        max_decoding_length=256, repetition_penalty=1.2, no_repeat_ngram_size=3,
    )
    hypothesis = results[0].hypotheses[0][1:]  # drop the language tag we forced
    return tokenizer.decode(
        tokenizer.convert_tokens_to_ids(hypothesis), skip_special_tokens=True
    )


def lug_to_en(text):
    return _translate(text, LUG, ENG)


# NLLB-600M knows little farming Luganda: "obuwunga" (powder) comes back as
# "flour", "kacungwa" (orange) as "pink", "emmwanyi" (coffee) as "skin". The
# terms the classifier depends on are matched by stem and appended in English.
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
    import re

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
    """NLLB's translation, followed by the glossary terms it may have lost."""
    english = lug_to_en(text)
    missing = [t for t in glossary_terms(text) if t not in english.lower()]
    return f"{english} ({', '.join(missing)})" if missing else english


def en_to_lug(text):
    return _translate(text, ENG, LUG)
