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
