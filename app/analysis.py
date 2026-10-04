"""The seam between the backend and P3's AI chain.

app/ never imports the heavy parts of ai/ at module level: the chain loads
NLLB-200 (CTranslate2, ~600 MB), e5-small and torch, which would sink both the
tests and the boot. The real chain is imported inside the function, only when
USE_REAL_AI=1; ai/lang.py, pure Python, is the exception.
"""

import logging
import os
import re

log = logging.getLogger(__name__)

# Advice template per label. Anything not in this map is not something we answer.
LABEL_TEMPLATE = {
    "healthy": "adv_healthy",
    "leaf_rust": "adv_leaf_rust",
    "phoma": "adv_phoma",
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
# "bean" is deliberately absent: NLLB renders plain coffee talk as "my coffee
# beans", including messages that were about leaves, so treating it as out of
# scope would hand every translated message to an agent untouched.
OUT_OF_SCOPE = (
    "berry", "berries", "cherry", "cherries",
    "branch", "branches", "twig", "twigs", "stem", "stems", "trunk", "root", "roots",
    "insect", "insects", "bug", "bugs", "ant", "ants", "borer", "beetle",
    "caterpillar", "weevil", "worm", "mealybug", "pest", "pests",
)


DECISIONS = ("answer", "clarify", "escalate")

# Filled by preload(), read by /health.
_STATE = {"enabled": False, "loaded": False}


def enabled():
    return os.environ.get("USE_REAL_AI") == "1"


def state():
    # enabled is read live: the preload runs in a thread and may not have
    # started when the first /health arrives.
    return {**_STATE, "enabled": enabled()}


def preload():
    """Load the heavy models once at server startup instead of on the first
    farmer's message.

    Never raises. A missing model is a degraded demo; a boot crash is no demo.
    """
    _STATE.clear()
    _STATE.update(loaded=False)
    if not enabled():
        return state()
    _STATE.update(loading=True)
    try:
        from ai.analyze import preload as load_chain  # lazy on purpose, see module docstring

        _STATE.update(loaded=True, seconds=round(load_chain(), 1))
    except Exception as exc:
        log.exception("the AI chain could not be preloaded")
        _STATE.update(error=f"{type(exc).__name__}: {exc}"[:200])
    finally:
        _STATE.update(loading=False)
    return state()


def analyze(text, clarify_answer=None):
    """-> {lang, text_en, label, proba, llm_label, decision, template_id, reason}

    decision is one of answer | clarify | escalate. template_id is the key to
    read out of templates.json, whatever the decision.
    """
    if not enabled():
        return _keyword_analyze(text, clarify_answer)
    try:
        from ai.analyze import analyze as real  # lazy on purpose, see module docstring

        return _checked(real(text, clarify_answer), text)
    except Exception:
        log.exception("the AI chain raised; handing this farmer to a human")
        return _result(text, "other", 0.0, "escalate", "unsure", "the AI chain failed")


def _checked(result, text):
    """A half-filled result must reach a human, not send a blank reply.

    P3's chain is still a stub that returns empty strings for every field;
    without this they would land as a reply with no template and no alert.
    """
    if (
        not isinstance(result, dict)
        or result.get("decision") not in DECISIONS
        or not result.get("template_id")
    ):
        log.warning("the AI chain returned an unusable result: %r", result)
        return _result(text, "other", 0.0, "escalate", "unsure", "unusable chain output")
    return result


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
    from ai.lang import detect_lang  # pure Python, nothing heavy

    return {
        "lang": detect_lang(text_en),
        "text_en": text_en,
        "label": label,
        "proba": proba,
        "llm_label": label,
        "decision": decision,
        "template_id": template_id,
        "reason": reason,
    }
