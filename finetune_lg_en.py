"""Fine-tuning d'un petit traducteur luganda -> anglais (Opus-MT, ~77 M de paramètres).

Version finale, telle qu'elle a réellement tourné pendant le hackathon (Colab, GPU T4, transformers v5).
Données : SALT (Sunbird, CC BY-SA 4.0) + SMS agricoles synthétiques (anglais généré par Claude,
traduit en luganda par l'API Sunbird).
Sortie : modèle CTranslate2 int8 (~80 Mo) publié sur https://huggingface.co/Adom4600/opus-lg-en-coffee-ct2

À exécuter section par section (une cellule Colab par section) :
    !pip install -q transformers datasets sentencepiece sacrebleu ctranslate2 huggingface_hub accelerate requests
Secrets : SUNBIRD_TOKEN (section 2), un token Hugging Face "Write" (section 6).
"""
import json
import os
import random
import subprocess
import time
from pathlib import Path

import requests
import sacrebleu
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

BASE = "Helsinki-NLP/opus-mt-lg-en"
HF_REPO = "Adom4600/opus-lg-en-coffee-ct2"
SUNBIRD_TOKEN = os.environ.get("SUNBIRD_TOKEN", "")
DOMAIN_EN = "sms_en.jsonl"                 # sortie de scripts/gen_sms_en.py
CACHE = Path("domain_pairs.jsonl")         # paires {"en", "lg"} déjà traduites (reprise possible)
MAX_DOMAIN = 200                           # nouveaux appels Sunbird max (0 = cache seulement)
PAUSE = 2.0                                # secondes entre deux appels (l'API renvoie 429 en parallèle)
DRIVE = "/content/drive/MyDrive"           # sauvegarde hors du runtime Colab si le Drive est monté
OUT = f"{DRIVE}/opus-lg-en-coffee" if os.path.isdir(DRIVE) else "opus-lg-en-coffee"
CT2 = OUT + "-ct2"
random.seed(0)

# ---------------------------------------------------------------- 1. SALT
salt = load_dataset("Sunbird/salt", "text-all")
DEV = "dev" if "dev" in salt else "validation"
# SALT a deux colonnes anglaises : vérifier laquelle est alignée avec lug_text.
for r in salt["train"].select(range(3)):
    print("SRC:", r.get("eng_source_text"), "\nTGT:", r.get("eng_target_text"), "\nLUG:", r["lug_text"], "\n---")
EN_COL, LG_COL = "eng_source_text", "lug_text"   # colonne utilisée pour le modèle publié


def to_pairs(ds):
    ds = ds.filter(lambda r: r[EN_COL] and r[LG_COL])
    return ds.map(lambda r: {"src": r[LG_COL], "tgt": r[EN_COL]}, remove_columns=ds.column_names)


salt_train, salt_dev, salt_test = to_pairs(salt["train"]), to_pairs(salt[DEV]), to_pairs(salt["test"])

# ---------------------------------------------------------------- 2. Données du domaine
def sunbird(text, src="eng", tgt="lug"):
    """Une requête à la fois, avec attente progressive sur 429."""
    wait = 10
    for _ in range(6):
        try:
            r = requests.post("https://api.sunbird.ai/tasks/translate",
                              headers={"Authorization": f"Bearer {SUNBIRD_TOKEN}"},
                              json={"source_language": src, "target_language": tgt, "text": text},
                              timeout=60)
        except requests.RequestException as exc:
            print("réseau :", exc); time.sleep(wait); wait = min(wait * 2, 120); continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            w = int(ra) if ra and ra.isdigit() else wait
            print(f"429 → pause {w}s"); time.sleep(w); wait = min(wait * 2, 120); continue
        if r.ok:
            return r.json()["output"]["translated_text"]
        print(r.status_code, r.text[:200]); return None
    return None


done = {}
if CACHE.exists():
    for line in CACHE.open():
        d = json.loads(line); done[d["en"]] = d["lg"]
english = [json.loads(line)["text"] for line in open(DOMAIN_EN)]
todo = [t for t in dict.fromkeys(english) if t not in done][:MAX_DOMAIN]
print(f"{len(done)} paires en cache, {len(todo)} à traduire")

with CACHE.open("a") as f:
    for i, en in enumerate(todo):
        lg = sunbird(en)
        time.sleep(PAUSE)
        if lg:
            done[en] = lg
            f.write(json.dumps({"en": en, "lg": lg}, ensure_ascii=False) + "\n"); f.flush()
        if i % 20 == 0:
            print(i, "/", len(todo), "| en cache :", len(done))

pairs = [{"src": lg, "tgt": en} for en, lg in done.items()]
random.shuffle(pairs)
n_test = min(50, len(pairs) // 4)
dom_test = Dataset.from_list(pairs[:n_test]) if n_test else None
dom_train = Dataset.from_list(pairs[n_test:]) if len(pairs) > n_test else None
print(f"SALT train {len(salt_train)} | domaine train {len(pairs) - n_test} | domaine test {n_test}")

# Le domaine est petit : répété 3 fois pour peser face à SALT.
train = concatenate_datasets([salt_train] + ([dom_train] * 3 if dom_train else [])).shuffle(seed=0)

# ---------------------------------------------------------------- 3. Entraînement (~15 min sur T4)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)


def prep(b):
    return tok(b["src"], text_target=b["tgt"], max_length=128, truncation=True)


train_tok = train.map(prep, batched=True, remove_columns=train.column_names)
dev_tok = salt_dev.map(prep, batched=True, remove_columns=salt_dev.column_names)

args = Seq2SeqTrainingArguments(          # transformers v5 : warmup_steps (warmup_ratio n'existe plus)
    output_dir="/content/ckpt", learning_rate=5e-5,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    num_train_epochs=3, warmup_steps=100, weight_decay=0.01, fp16=True,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=50, report_to="none")
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_tok, eval_dataset=dev_tok,
                         data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok)
trainer.train()
trainer.save_model(OUT); tok.save_pretrained(OUT)


# ---------------------------------------------------------------- 4. Évaluation avant / après
def evaluate(model_dir, ds, n=500, bs=32):
    t = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda").half().eval()
    ds = ds.select(range(min(n, len(ds))))
    hyps = []
    for i in range(0, len(ds), bs):
        batch = t(ds["src"][i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            out = m.generate(**batch, num_beams=2, max_length=128)
        hyps += t.batch_decode(out, skip_special_tokens=True)
    refs = [ds["tgt"]]
    return {"chrF": round(sacrebleu.corpus_chrf(hyps, refs).score, 1),
            "BLEU": round(sacrebleu.corpus_bleu(hyps, refs).score, 1)}


res = {"salt_test_before": evaluate(BASE, salt_test), "salt_test_after": evaluate(OUT, salt_test)}
if dom_test:
    res["domain_test_before"] = evaluate(BASE, dom_test)
    res["domain_test_after"] = evaluate(OUT, dom_test)
res.update(n_salt_train=len(salt_train), n_domain_train=len(dom_train) if dom_train else 0)
json.dump(res, open(f"{OUT}/scores.json", "w"), indent=2)
print(json.dumps(res, indent=2))

# ---------------------------------------------------------------- 5. Export CTranslate2 int8 (~80 Mo)
subprocess.run(["ct2-transformers-converter", "--model", OUT, "--output_dir", CT2,
                "--quantization", "int8", "--force"], check=True)
tok.save_pretrained(CT2)                   # source.spm, target.spm, vocab.json, config du tokenizer

# ---------------------------------------------------------------- 6. Publication Hugging Face
from huggingface_hub import HfApi, login  # noqa: E402

login()
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(folder_path=CT2, repo_id=HF_REPO)
print(api.list_repo_files(HF_REPO))