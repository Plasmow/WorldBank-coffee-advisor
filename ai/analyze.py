"""AI chain: translation -> classifier -> LLM second opinion -> decision.

Contract (P3 -> P2):
    analyze(text, clarify_answer=None) -> {
        "lang", "text_en", "label", "proba", "llm_label",
        "decision",     # "answer" | "clarify" | "escalate"
        "template_id",  # key of data/templates.json
        "reason",
    }

Rules:
- answer only if proba >= threshold AND the LLM gives the same label
  (LLM unreachable: the calibrated classifier alone, plan B of the roadmap);
- "other" never gets advice: straight to the agent when the classifier is
  sure and the LLM agrees (or is down), otherwise first the clarifying question
  -- a vague or badly translated message deserves one question;
- otherwise one clarifying question, chosen by the LLM among fixed templates,
  then the agent.
"""
from ai import translate
from ai.classifier import THRESHOLD, classify
from ai.lang import detect_lang
from ai.llm import llm_predict

ADVICE_TEMPLATES = {
    "leaf_rust": "adv_leaf_rust",
    "phoma": "adv_phoma",
    "healthy": "adv_healthy",
}
CLARIFY_TEMPLATE = "clarify"
ESCALATE_TEMPLATE = "unsure"


def _to_english(text: str, lang: str) -> str:
    if lang == "en" or not any(c.isalpha() for c in text):
        return text
    try:
        return translate.lug_to_en_with_terms(text)
    except Exception:
        return text  # worst case we classify the raw text; the threshold sorts it out


def analyze(text: str, clarify_answer: str | None = None) -> dict:
    lang = detect_lang(text, default="lg")
    text_en = _to_english(text, lang)
    label, proba = None, 0.0
    if clarify_answer:
        # The answer to the clarifying question completes the first message.
        answer_lang = detect_lang(clarify_answer, default=lang)
        answer_en = _to_english(clarify_answer, answer_lang)
        text_en = f"{text_en}. {answer_en}"
        lang = answer_lang
        # The answer alone is often the clearer of the two: the first message
        # was vague, that is why we asked. Keep it if it is a confident diagnosis.
        label, proba = classify(answer_en)

    combined = classify(text_en)
    if label in (None, "other") or proba < THRESHOLD or combined[1] >= proba:
        label, proba = combined
    llm = llm_predict(text_en)
    llm_label = llm["label"]

    confident = proba >= THRESHOLD

    if label == "other":
        # Never advice. Straight to the agent when both models are sure (or
        # the LLM is down); otherwise one question first.
        llm_sure_other = llm_label == "other" and llm.get("sure", True)
        if clarify_answer or (confident and (llm_label is None or llm_sure_other)):
            decision, reason = "escalate", "other"
        else:
            decision, reason = "clarify", "unclear"
    elif confident and llm_label == label:
        decision, reason = "answer", "agree"
    elif confident and llm_label is None:
        decision, reason = "answer", "llm_unavailable"
    else:
        reason = "low_confidence" if not confident else "disagree"
        decision = "escalate" if clarify_answer else "clarify"

    template_id = {
        "answer": ADVICE_TEMPLATES.get(label),
        "clarify": llm.get("question") or CLARIFY_TEMPLATE,
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


def preload() -> float:
    """Load every model once at server startup. Returns seconds spent."""
    import time

    from ai import classifier, llm

    started = time.monotonic()
    translate.load()
    classifier.preload()
    llm.preload()
    return time.monotonic() - started


if __name__ == "__main__":
    import time

    for msg, answer in [
        ("orange powder under the leaves", None),
        ("Ebikoola by'emmwanyi zange birina obuwunga obwa langi ya kacungwa wansi waabyo", None),
        ("my coffee looks strange", None),
        ("my coffee looks strange", "orange powder under the leaves"),
    ]:
        t0 = time.perf_counter()
        print(analyze(msg, answer), f"{time.perf_counter() - t0:.1f}s")
