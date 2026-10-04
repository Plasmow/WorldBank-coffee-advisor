from functools import lru_cache
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

from time import time

MODEL_NAME = "facebook/nllb-200-distilled-600M"  # uniquement pour les tokenizers
CT2_DIR = Path(__file__).resolve().parent / "nllb-600m-ct2"

LUG = "lug_Latn"
ENG = "eng_Latn"


@lru_cache(maxsize=1)
def get_translator() -> ctranslate2.Translator:
    return ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8")


@lru_cache(maxsize=None)
def get_tokenizer(src_lang: str):
    return AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_lang)


def preload():
    """À appeler au démarrage du serveur : charge le modèle et les tokenizers.

    Une traduction factice absorbe la lenteur du premier appel (~10 s).
    """
    start = time()
    
    get_translator()
    get_tokenizer(LUG)
    get_tokenizer(ENG)
    lug_to_en("Ebikoola bya kawa")

    return time() - start

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