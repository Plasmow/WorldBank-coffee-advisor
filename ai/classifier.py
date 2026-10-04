"""Classifieur FACTICE (mots-clés) en attendant la vraie version de P1.

Interface attendue par analyze.py :
    THRESHOLD: float
    classify(text_en: str) -> tuple[str, float]   # (label, proba)
    preload() -> None
"""

THRESHOLD = 0.8

KEYWORDS = {
    "leaf_rust": ["orange", "powder", "dust", "rust"],
    "phoma": ["black", "dark brown", "tip", "dieback", "dying back"],
    "healthy": ["green", "shiny", "glossy", "fine", "good"],
}


def preload() -> None:
    pass


def classify(text_en: str) -> tuple[str, float]:
    t = text_en.lower()
    scores = {label: sum(k in t for k in kws) for label, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other", 0.5
    return best, 0.85 if scores[best] >= 2 else 0.6
