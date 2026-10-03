"""The seam between the backend and P3's AI chain.

app/ never imports ai/ at module level: ai/translate.py loads NLLB-200 (~2.4 GB)
on import, which would sink both the tests and the Replit boot. The real chain
is imported inside the function, only when USE_REAL_AI=1.
"""

import os

# P3's templates.json uses capitalised keys; the contract labels are snake_case.
# Bridging here beats renaming someone else's file mid-hackathon.
LABEL_TEMPLATE = {
    "healthy": "Healthy",
    "leaf_rust": "Rust",
    "phoma": "Phoma",
    "other": "Unknown",
}

HINTS = {
    "leaf_rust": ("rust", "orange", "yellow spot", "yellow powder", "powder", "underside"),
    "phoma": ("phoma", "black spot", "dark spot", "brown lesion", "dieback", "necrosis", "lesion"),
    "healthy": ("healthy", "no problem", "looks good", "looks fine", "all good"),
}
# Words that say "something is wrong" without saying what: worth one question.
VAGUE = ("leaf", "leaves", "plant", "tree", "coffee", "berries", "sick", "dying", "problem", "bad")


def analyze(text, clarify_answer=None):
    """-> {lang, text_en, label, proba, llm_label, decision, template_id}

    decision is one of answer | clarify | escalate.
    """
    if os.environ.get("USE_REAL_AI") == "1":
        from ai.analyze import analyze as real  # lazy on purpose, see module docstring

        return real(text, clarify_answer)
    return _keyword_analyze(text, clarify_answer)


def _keyword_analyze(text, clarify_answer=None):
    blob = f"{text} {clarify_answer or ''}".lower()
    scores = {
        label: sum(1 for w in words if w in blob) for label, words in HINTS.items()
    }
    label, hits = max(scores.items(), key=lambda kv: kv[1])

    if hits:
        decision, proba = "answer", 0.9
    elif clarify_answer is not None:
        # We already asked our one question and still cannot tell.
        label, decision, proba = "other", "escalate", 0.0
    elif any(w in blob for w in VAGUE):
        label, decision, proba = "other", "clarify", 0.0
    else:
        label, decision, proba = "other", "escalate", 0.0

    return {
        "lang": "en",
        "text_en": text,
        "label": label,
        "proba": proba,
        "llm_label": label,
        "decision": decision,
        "template_id": LABEL_TEMPLATE[label],
    }
