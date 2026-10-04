"""The real models, end to end. Slow (~1 min to load), so opt-in:

    RUN_MODEL_TESTS=1 python -m pytest tests/test_models.py -v

Downloads the Opus-MT CTranslate2 model on first run (~80 MB), and needs
torch + transformers for e5-small, optionally Ollama for the LLM.
Run it before every deploy that touches ai/ or model/.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load the real models"
)


def test_the_model_translates_luganda_to_english():
    from ai import translate

    out = translate.lug_to_en("Ebikoola by'emmwanyi zange birina obuwunga wansi waabyo")
    assert out and out.isascii()
    assert out != "Ebikoola by'emmwanyi zange birina obuwunga wansi waabyo"


def test_the_classifier_recognises_a_textbook_rust_description():
    from ai.classifier import THRESHOLD, classify

    label, proba = classify("orange powder under the leaves")
    assert label == "leaf_rust" and proba >= THRESHOLD


def test_the_classifier_sends_berries_to_other():
    from ai.classifier import classify

    assert classify("small holes in the berries")[0] == "other"


@pytest.mark.parametrize("text", [
    "orange powder under the leaves",
    "Ebikoola by'emmwanyi zange birina obuwunga obwa langi ya kacungwa wansi waabyo",
    "my coffee looks strange",
    "small holes in the berries",
])
def test_the_full_chain_returns_a_usable_decision(text):
    from ai.analyze import analyze

    r = analyze(text)
    assert r["decision"] in ("answer", "clarify", "escalate")
    assert r["template_id"]
    if r["label"] == "other":
        assert r["decision"] != "answer"


def test_vague_message_is_never_answered():
    from ai.analyze import analyze

    assert analyze("my coffee looks strange")["decision"] != "answer"


# The classifier was once trained only on photo captions for the three disease
# classes, with 'other' as the only class written like an SMS. It learned the
# register, not the disease: real messages scored other=1.00, phoma recall on
# held-out SMS was 3%. These guard the fix.

SMS_REGISTER = [
    ("My coffee leaves have yellow powder underneath. Trees losing leaves fast.", "leaf_rust"),
    ("Orange dust on coffee leaves. Started two weeks ago. Is this normal?", "leaf_rust"),
    ("My coffee leaves have dark brown spots. What should I do?", "phoma"),
    ("Help please. Coffee leaves turning black after last week cold nights.", "phoma"),
    ("Good morning. What is the price of parchment coffee today?", "other"),
]


@pytest.mark.parametrize("text,expected", SMS_REGISTER)
def test_the_classifier_reads_sms_register_not_just_photo_captions(text, expected):
    from ai.classifier import classify

    label, proba = classify(text)
    assert label == expected, f"{text!r} -> {label} ({proba:.2f})"


def test_held_out_sms_are_not_all_swept_into_other():
    """Recall on the SMS the model never saw. It used to be 3% for phoma."""
    import collections
    import json
    import pathlib

    from ai.classifier import classify

    rows = [json.loads(l) for l in
            pathlib.Path("data/synthetic/dataset.jsonl").read_text().splitlines() if l.strip()]
    held_out = [(r["text"], r["label"]) for r in rows
                if r.get("kind") == "sms" and r["split"] == "test"]
    assert len(held_out) >= 30, "the SMS never reached the dataset"

    hits = collections.Counter()
    totals = collections.Counter()
    for text, truth in held_out:
        totals[truth] += 1
        hits[truth] += classify(text)[0] == truth

    for label in ("leaf_rust", "phoma", "healthy"):
        if totals[label]:
            recall = hits[label] / totals[label]
            assert recall >= 0.7, f"{label} recall {recall:.0%} on held-out SMS"
