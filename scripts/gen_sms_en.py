"""Génère des SMS d'agricultrice en anglais avec Claude, pour entraîner le traducteur luganda -> anglais.

Sortie : data/synthetic/sms_en.jsonl, une ligne par SMS : {"text": "...", "topic": "..."}
Ces textes sont SYNTHÉTIQUES (générés par Claude) : à déclarer comme tels dans docs/datasheet.md.

Lancer depuis la racine du dépôt :
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/gen_sms_en.py --n 300
Relancer le script complète le fichier sans dupliquer ce qui existe déjà.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import anthropic

OUT = Path("data/synthetic/sms_en.jsonl")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")   # vérifier l'identifiant exact dans la console Anthropic

# Thèmes couverts : les symptômes (cœur du produit) et le vocabulaire agricole courant,
# pour que le traducteur ne connaisse pas seulement les maladies des feuilles.
TOPICS = {
    "orange or yellow powder under coffee leaves, leaves falling (do not name the disease)": 3,
    "dark brown or black spots on coffee leaves, leaves drying after cold nights (do not name the disease)": 3,
    "coffee leaves that look green and healthy, but low harvest or other worries": 2,
    "problems on coffee berries or branches: holes in berries, black berries, dry branches, insects": 2,
    "vague messages saying something is wrong with the coffee trees, without details": 2,
    "questions about coffee prices, selling parchment, the buyer, the cooperative": 1,
    "weather, rain, drying coffee, harvest time, weeding, mulching": 1,
    "replies to an advisor: better, the same, worse, thank you, asking for an agent to visit": 1,
}

PROMPT = """You write realistic SMS messages sent by smallholder coffee farmers on Mt Elgon, Uganda,
to a free SMS advisory service. Topic: {topic}.

Write {k} different messages. Rules:
- Plain, simple, correct English (they will be translated into Luganda, so no spelling mistakes or slang).
- Short: 4 to 25 words, never more than 150 characters.
- Vary the style: some very short, some with context (how many trees, since when, recent weather, which part of the farm).
- Describe what the farmer sees; do not use scientific or disease names.
Return only a JSON array of strings."""


def ask(client, topic, k):
    msg = client.messages.create(
        model=MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(topic=topic, k=k)}],
    )
    text = msg.content[0].text
    match = re.search(r"\[.*\]", text, re.S)                 # tolère un texte autour du JSON
    return [s.strip() for s in json.loads(match.group(0)) if isinstance(s, str)] if match else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="nombre total de SMS visé")
    parser.add_argument("--batch", type=int, default=20, help="SMS demandés par appel")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["text"].lower() for l in OUT.open()} if OUT.exists() else set()
    client = anthropic.Anthropic()
    weights = list(TOPICS.values())
    print(f"{len(seen)} SMS déjà présents, objectif {args.n}")

    with OUT.open("a") as f:
        tries = 0
        while len(seen) < args.n and tries < args.n:          # garde-fou contre une boucle infinie
            tries += 1
            topic = random.choices(list(TOPICS), weights=weights)[0]
            try:
                batch = ask(client, topic, args.batch)
            except Exception as exc:
                print("erreur API, on continue :", exc)
                continue
            added = 0
            for s in batch:
                if 3 <= len(s.split()) <= 30 and len(s) <= 160 and s.lower() not in seen:
                    seen.add(s.lower())
                    f.write(json.dumps({"text": s, "topic": topic.split(",")[0]}, ensure_ascii=False) + "\n")
                    added += 1
            print(f"+{added} ({len(seen)}/{args.n}) · {topic[:50]}")

    print(f"Terminé : {OUT} ({len(seen)} SMS)")


if __name__ == "__main__":
    main()