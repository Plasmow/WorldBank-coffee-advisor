"""Chaîne IA : traduction -> classifieur -> second avis LLM -> décision.

Contrat (P3 -> P2) :
    analyze(text, clarify_answer=None) -> {
        "lang", "text_en", "label", "proba", "llm_label",
        "decision",     # "answer" | "clarify" | "escalate"
        "template_id",  # clé de data/templates.json
        "reason",
    }

Règles :
- on répond seulement si proba >= seuil ET le LLM donne le même label ;
- la classe "other" ne reçoit jamais de conseil : agent direct si le classifieur
  est sûr, sinon d'abord la question de précision (message flou) ;
- sinon une seule question de précision, puis transmission à l'agent.
"""
import re

from ai.classifier import THRESHOLD, classify
from ai.llm import llm_predict
from ai.translate import lug_to_en

# label -> clé de data/templates.json
ADVICE_TEMPLATES = {
    "leaf_rust": "Rust",
    "phoma": "Phoma",
    "healthy": "Healthy",
}
CLARIFY_TEMPLATE = "clarify"
ESCALATE_TEMPLATE = "unsure"


# ---------- détection de langue ----------

EN_WORDS = {
    "the", "a", "an", "is", "are", "my", "on", "of", "and", "with", "leaf",
    "leaves", "coffee", "spots", "spot", "yellow", "orange", "brown", "black",
    "powder", "tree", "trees", "plant", "plants", "have", "has", "there",
    "some", "under", "it", "they", "what", "green", "dry", "dying", "turning",
}


def detect_lang(text: str) -> str:
    """Heuristique simple : part de mots anglais courants -> "en", sinon "lg"."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "lg"
    en_ratio = sum(w in EN_WORDS for w in words) / len(words)
    return "en" if en_ratio >= 0.3 else "lg"


# ---------- décision ----------

def _translate(text: str, lang: str) -> str:
    if lang == "en":
        return text
    try:
        return lug_to_en(text)
    except Exception:
        return text  # au pire on classe le texte brut ; le seuil fera le tri


def analyze(text: str, clarify_answer: str | None = None) -> dict:
    lang = detect_lang(text)
    text_en = _translate(text, lang)
    if clarify_answer:
        # la réponse à la question de précision complète le premier message
        text_en = f"{text_en}. {_translate(clarify_answer, detect_lang(clarify_answer))}"

    label, proba = classify(text_en)
    llm = llm_predict(text_en)
    llm_label = llm["label"]

    confident = proba >= THRESHOLD

    if label == "other" and (confident or clarify_answer):
        decision, reason = "escalate", "other"
    elif confident and llm_label == label:
        decision, reason = "answer", "agree"
    elif confident and llm_label is None:
        # Ollama indisponible : plan B de la feuille de route, classifieur calibré seul
        decision, reason = "answer", "llm_unavailable"
    else:
        reason = "low_confidence" if not confident else "disagree"
        decision = "escalate" if clarify_answer else "clarify"

    template_id = {
        "answer": ADVICE_TEMPLATES.get(label),
        "clarify": CLARIFY_TEMPLATE,
        "escalate": ESCALATE_TEMPLATE,
    }[decision]

    return {
        "lang": lang,
        "text_en": text_en,
        "label": label,
        "proba": round(proba, 3),
        "llm_label": llm_label,
        "decision": decision,
        "template_id": template_id,
        "reason": reason,
    }


if __name__ == "__main__":
    import time

    for msg, answer in [
        ("orange powder under the leaves", None),
        ("Ebikoola bya kawa birina obutonnyeze obwa kyenvu", None),
        ("my coffee looks strange", None),
        ("my coffee looks strange", "black marks at the tips of the young leaves"),
    ]:
        t0 = time.perf_counter()
        print(analyze(msg, answer), f"{time.perf_counter() - t0:.1f}s")
