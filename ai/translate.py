"""Luganda <-> English, with CTranslate2 int8.

Why not transformers at runtime. NLLB-200-distilled-600M in float32 is ~2.4 GB
of weights and needs torch, another ~800 MB installed. Converted once to
CTranslate2 int8 the same model is ~600 MB, loads in seconds, runs several
times faster on CPU, and the runtime no longer needs torch at all -- only the
tokenizer. That is the difference between fitting on Replit and not.

Convert once, on a laptop:  python scripts/convert_nllb_ct2.py
Then preload at boot:       app.analysis.preload(), called from the lifespan.
"""

import logging
import os
import pathlib
import time
from functools import lru_cache

log = logging.getLogger(__name__)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_DIR = "model/nllb-ct2-int8"
LUG, ENG = "lug_Latn", "eng_Latn"


def model_dir():
    # Read at call time, not import time: tests and Replit set this after import.
    return pathlib.Path(os.environ.get("NLLB_CT2_DIR") or DEFAULT_DIR)


def _ensure_model():
    path = model_dir()
    if (path / "model.bin").exists():
        return path

    # Weights never go in git, so Replit pulls them once from the Hub instead.
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
    path = _ensure_model()
    import ctranslate2  # after the model check: the slim install has no ctranslate2
    compute_type = os.environ.get("NLLB_COMPUTE_TYPE") or "int8"
    log.info("loading CTranslate2 model from %s (%s)", path, compute_type)
    return ctranslate2.Translator(str(path), device="cpu", compute_type=compute_type)


@lru_cache(maxsize=1)
def get_tokenizer():
    """One tokenizer for both directions.

    NLLB uses the same 256k vocabulary whichever way you translate; only the
    source language tag changes, and that is a settable attribute. Caching one
    per language would hold a second copy of that vocabulary for ~470 MB of
    nothing.
    """
    from transformers import AutoTokenizer

    # The converter copies the tokenizer next to the weights, so it can never
    # drift from the model it was converted from.
    return AutoTokenizer.from_pretrained(str(model_dir()))


def load():
    """Warm everything up. Called once at server startup; returns seconds."""
    started = time.monotonic()
    get_translator()
    get_tokenizer()
    elapsed = time.monotonic() - started
    log.info("translation model ready in %.1fs", elapsed)
    return elapsed


def is_loaded():
    return get_translator.cache_info().currsize > 0


def _translate(text, src_lang, tgt_lang):
    text = (text or "").strip()
    if not text:
        return ""

    tokenizer = get_tokenizer()
    tokenizer.src_lang = src_lang
    translator = get_translator()

    # CTranslate2 works on tokens, not ids. The source already carries its
    # language tag from the tokenizer; the target one is forced as a prefix.
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch([source], target_prefix=[[tgt_lang]])
    hypothesis = results[0].hypotheses[0][1:]  # drop the language tag we forced
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(hypothesis))


def lug_to_en(text):
    return _translate(text, LUG, ENG)


def en_to_lug(text):
    return _translate(text, ENG, LUG)
