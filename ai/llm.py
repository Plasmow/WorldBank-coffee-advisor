"""Second avis : qwen2.5:3b via Ollama, sortie JSON imposée, température 0.

Le LLM ne rédige jamais de texte pour Noor : il choisit seulement un label
parmi les 4 classes, que analyze() compare au label du classifieur.
"""
import json
import os
from functools import lru_cache

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", "10"))

LABELS = ["leaf_rust", "phoma", "healthy", "other"]

SYSTEM_PROMPT = """You classify short SMS messages sent by smallholder arabica coffee farmers on Mount Elgon, Uganda.
The farmer describes what they see on their coffee plants. Pick exactly one label:

- leaf_rust: orange or yellow powdery/dusty spots, usually on the underside of leaves; pale yellow patches on the upper side; leaves falling early.
- phoma: dark brown to black patches or spots, often starting at leaf tips or edges, sometimes with a lighter centre; young leaves and shoot tips turning black and dying back, often in cold, wet weather.
- healthy: leaves described as green, glossy, uniform, without spots, patches or powder.
- other: anything else - insects, holes, tunnels or trails in leaves, problems on berries, stems or roots, wilting of the whole tree, prices, greetings, unrelated or too vague messages.

If the message is unclear or does not match one description well, answer "other".
Answer only with JSON: {"label": "<label>", "reason": "<3 to 6 words>"}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}


@lru_cache(maxsize=1)
def get_client():
    # Imported here, not at module level: without the package llm_predict
    # catches the error and returns label=None, which is exactly the
    # "Ollama indisponible" fallback documented below.
    import ollama

    # OLLAMA_HOST est lu automatiquement par le client
    return ollama.Client(timeout=TIMEOUT_S)


def preload() -> None:
    """À appeler au démarrage du serveur : charge le modèle et met le prompt système en cache.

    Le premier appel à froid peut dépasser TIMEOUT_S ; on l'ignore ici.
    """
    llm_predict("warm-up")


def llm_predict(text_en: str) -> dict:
    """Renvoie {"label": str | None, "reason": str}.

    label vaut None si Ollama est injoignable ou répond hors schéma :
    analyze() retombe alors sur le classifieur seul.
    """
    try:
        response = get_client().chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_en},
            ],
            format=SCHEMA,
            options={"temperature": 0, "num_predict": 40},
            keep_alive="1h",
        )
        result = json.loads(response["message"]["content"])
    except Exception as e:  # Ollama arrêté, timeout, JSON invalide...
        return {"label": None, "reason": f"llm_error: {type(e).__name__}"}

    label = result.get("label")
    if label not in LABELS:
        return {"label": None, "reason": f"llm_bad_label: {label!r}"}
    return {"label": label, "reason": str(result.get("reason", ""))}


if __name__ == "__main__":
    for msg in [
        "orange powder under the leaves",
        "the tips of the young leaves are turning black and drying",
        "my leaves are dark green and shiny",
        "what is the price of coffee today?",
    ]:
        print(msg, "->", llm_predict(msg))
