"""The seam between the backend and P3's AI chain.

app/ never imports ai/ at module level: ai/translate.py loads NLLB-200 (~2.4 GB)
on import, which would sink both the tests and the Replit boot. The real chain
is imported inside the function, only when USE_REAL_AI=1.
"""

import os
import re

# P3's templates.json uses capitalised keys for the three diagnoses; the
# contract labels are snake_case. Bridging here beats renaming someone else's
# file mid-hackathon. Anything not in this map is not something we answer.
LABEL_TEMPLATE = {
    "healthy": "Healthy",
    "leaf_rust": "Rust",
    "phoma": "Phoma",
}

HINTS = {
    "leaf_rust": ("rust", "orange", "yellow spot", "yellow powder", "powder", "underside"),
    "phoma": ("phoma", "black spot", "dark spot", "brown spot", "brown lesion",
              "dieback", "necrosis", "lesion"),
    "healthy": ("healthy", "no problem", "looks good", "looks fine", "all good"),
}

# Understood perfectly well, and still none of our business: the classifier was
# trained on leaves. Asking a clarifying question here would burn the farmer's
# one reply to arrive at the same place, so this goes straight to a human.
OUT_OF_SCOPE = (
    "berry", "berries", "cherry", "cherries", "bean", "beans",
    "branch", "branches", "twig", "twigs", "stem", "stems", "trunk", "root", "roots",
    "insect", "insects", "bug", "bugs", "ant", "ants", "borer", "beetle",
    "caterpillar", "weevil", "worm", "mealybug", "pest", "pests",
)


def analyze(text, clarify_answer=None):
    """-> {lang, text_en, label, proba, llm_label, decision, template_id, reason}

    decision is one of answer | clarify | escalate. template_id is the key to
    read out of templates.json, whatever the decision.
    """
    if os.environ.get("USE_REAL_AI") == "1":
        from ai.analyze import analyze as real  # lazy on purpose, see module docstring

        return real(text, clarify_answer)
    return _keyword_analyze(text, clarify_answer)


def _hits(words, blob):
    """Whole words only, plural tolerated.

    Substring matching looks harmless until "ant" fires inside "plants" and a
    farmer describing her plants is shipped straight to a human.
    """
    return sum(1 for w in words if re.search(rf"\b{re.escape(w)}s?\b", blob))


def _keyword_analyze(text, clarify_answer=None):
    # Her answer is read together with what she first wrote: "orange powder"
    # only means something next to "on the leaves".
    full = f"{text} {clarify_answer}".strip() if clarify_answer else text
    blob = full.lower()

    scores = {label: _hits(words, blob) for label, words in HINTS.items()}
    label, hits = max(scores.items(), key=lambda kv: kv[1])
    if hits:
        return _result(full, label, 0.9, "answer", LABEL_TEMPLATE[label])

    if _hits(OUT_OF_SCOPE, blob):
        return _result(full, "other", 0.0, "escalate", "unsure", "outside the trained classes")

    if clarify_answer is None:
        return _result(full, "other", 0.0, "clarify", "clarify", "could not read the message")

    # We asked our one question and are none the wiser.
    return _result(full, "other", 0.0, "escalate", "unsure", "still unclear after one question")


def _result(text_en, label, proba, decision, template_id, reason=""):
    return {
        "lang": "en",
        "text_en": text_en,
        "label": label,
        "proba": proba,
        "llm_label": label,
        "decision": decision,
        "template_id": template_id,
        "reason": reason,
    }
