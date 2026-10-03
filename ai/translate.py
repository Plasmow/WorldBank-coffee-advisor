from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"

tokenizer_lug = AutoTokenizer.from_pretrained(model_name, src_lang="lug_Latn")
tokenizer_en = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def lug_to_en(input: str):
    inputs = tokenizer_lug(input, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer_en.convert_tokens_to_ids("eng_Latn"),
        max_length=512)
    
    return tokenizer_en.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def en_to_lug(input: str):
    inputs = tokenizer_en(input, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer_lug.convert_tokens_to_ids("lug_Latn"),
        max_length=512)
    
    return tokenizer_lug.batch_decode(translated_tokens, skip_special_tokens=True)[0]

