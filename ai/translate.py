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
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"  # uniquement pour les tokenizers
CT2_DIR = Path(__file__).resolve().parent / "nllb-600m-ct2"

LUG = "lug_Latn"
ENG = "eng_Latn"


@lru_cache(maxsize=1)
def get_translator():
    path = _ensure_model()
    import ctranslate2  # after the model check: the slim install has no ctranslate2
    compute_type = os.environ.get("NLLB_COMPUTE_TYPE") or "int8"
    log.info("loading CTranslate2 model from %s (%s)", path, compute_type)
    return ctranslate2.Translator(str(path), device="cpu", compute_type=compute_type)
def get_translator() -> ctranslate2.Translator:
    return ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8")


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


def preload() -> None:
    """À appeler au démarrage du serveur : charge le modèle et les tokenizers.

    Une traduction factice absorbe la lenteur du premier appel (~10 s).
    """
    get_translator()
    get_tokenizer(LUG)
    get_tokenizer(ENG)
    lug_to_en("Ebikoola bya kawa")


def _translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer = get_tokenizer(src_lang)
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    result = get_translator().translate_batch(
        [source],
        target_prefix=[[tgt_lang]],
        max_decoding_length=512,
    )
    target = result[0].hypotheses[0][1:]  # on retire le token de langue cible
    return tokenizer.decode(
        tokenizer.convert_tokens_to_ids(target), skip_special_tokens=True
    )


def lug_to_en(text: str) -> str:
    return _translate(text, LUG, ENG)


def en_to_lug(text: str) -> str:
    return _translate(text, ENG, LUG)
