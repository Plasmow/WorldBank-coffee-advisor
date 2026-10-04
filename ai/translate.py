from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"


@lru_cache(maxsize=1)
def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,  # limite le pic de RAM au chargement
    )
    model.eval()
    return model


@lru_cache(maxsize=None)
def get_tokenizer(src_lang: str):
    return AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_lang)


def _translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer = get_tokenizer(src_lang)
    model = get_model()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512,
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def lug_to_en(text: str) -> str:
    return _translate(text, "lug_Latn", "eng_Latn")


def en_to_lug(text: str) -> str:
    return _translate(text, "eng_Latn", "lug_Latn")