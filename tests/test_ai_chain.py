"""The real chain must be importable and usable without the heavy parts.

Nothing in ai/ may need a model at import time. The farmer writing in English
needs no translation, and a missing Ollama is a documented fallback -- the
classifier answers alone. Both were broken by a merge: ollama was imported at
module level, so the whole chain was unimportable without it.
"""

import pytest

CONTRACT = {"lang", "text_en", "label", "proba", "llm_label", "decision", "template_id", "reason"}
DECISIONS = {"answer", "clarify", "escalate"}


def test_the_chain_imports_without_ollama_or_a_model():
    from ai.analyze import analyze  # noqa: F401


def test_english_needs_no_translation_model():
    from ai.analyze import analyze

    r = analyze("orange powder under the leaves")
    assert set(r) >= CONTRACT
    assert r["lang"] == "en"
    assert r["decision"] in DECISIONS


def test_a_missing_llm_falls_back_to_the_classifier_alone():
    from ai.analyze import analyze

    r = analyze("orange powder and rust dust under the leaves")
    assert r["llm_label"] is None          # no daemon here
    assert r["decision"] == "answer"       # the documented plan B
    assert r["reason"] == "llm_unavailable"
    assert r["template_id"] == "Rust"


def test_translate_exposes_both_names_the_team_uses():
    import ai.translate as tr

    assert tr.load is tr.preload   # app/ calls load(), ai/ called it preload()


def test_translate_says_what_is_missing_instead_of_importing_a_model(monkeypatch, tmp_path):
    import ai.translate as tr

    monkeypatch.setenv("NLLB_CT2_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("NLLB_CT2_REPO", "")
    tr.get_translator.cache_clear()
    with pytest.raises(FileNotFoundError, match="convert_nllb_ct2"):
        tr.load()


def test_the_translator_is_cached(monkeypatch, tmp_path):
    # The merge left get_translator defined twice, the second one uncached:
    # every sentence would have reloaded 621 MB from disk.
    import ai.translate as tr

    assert hasattr(tr.get_translator, "cache_info")
