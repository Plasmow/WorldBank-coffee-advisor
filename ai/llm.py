"""Second opinion: qwen2.5:3b via Ollama, JSON schema enforced, temperature 0.

The LLM never writes text for Noor. It picks a label among the 4 classes,
which analyze() compares with the classifier's, and, for when the two are in
doubt, the id of the one clarifying question (a key of data/templates.json)
that would best settle it. The prompt shows the exact English of each
question, so the LLM knows what Noor will actually be asked.

OLLAMA_HOST points the client at any Ollama server (the laptop through a
tunnel, for a cloud host that cannot run one). Unreachable -> label None, and
analyze() falls back to the calibrated classifier alone (plan B of the roadmap).
"""
import json
import os
import pathlib
from functools import lru_cache

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", "30"))

LABELS = ["leaf_rust", "phoma", "healthy", "other"]

# Clarifying questions the LLM may choose from (keys of data/templates.json),
# with when each one helps. The question's own wording is read from the file.
QUESTION_USE = {
    "ask_plant_part": "the message does not say which part of the plant is affected",
    "ask_leaf_look": "the problem is on the leaves but the message does not say what the marks look like",
    "ask_powder_or_dark_patches": "the message mentions spots or patches on leaves that could be either leaf_rust or phoma",
    "ask_yellow_pattern": "the leaves are turning yellow, which may be leaf_rust or something else",
    "ask_black_tips_cold": "the message mentions black or dying leaf tips or shoots, to confirm phoma",
    "ask_any_marks": "the leaves may be healthy but it is not clear whether there are any marks",
    "clarify": "none of the questions above fits the message",
}
QUESTIONS = list(QUESTION_USE)
FALLBACK_QUESTION = "clarify"
TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "data" / "templates.json"


def _question_menu():
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    lines = [f'- {qid}: asks "{templates[qid]["en"]}" Use when {use}.'
             for qid, use in QUESTION_USE.items()]
    return "\n".join(lines)


SYSTEM_PROMPT = """You classify short SMS messages sent by smallholder arabica coffee farmers on Mount Elgon, Uganda.
The text is often a rough machine translation from Luganda. The farmer describes what they see on their coffee plants.

Pick exactly one label:
- leaf_rust: orange or yellow powdery/dusty spots, usually on the underside of leaves; pale yellow patches on the upper side; leaves falling early.
- phoma: dark brown, reddish-brown or black patches or spots on leaves, often at tips or edges, dry and papery; young leaves and shoot tips turning black and dying back, often in cold, wet weather.
- healthy: leaves described as green, glossy, uniform, without spots, patches or powder.
- other: anything else - insects, holes, problems on berries, stems or roots, wilting of the whole tree, prices, greetings, unrelated or too vague messages.

If the message is unclear or does not match one description well, answer "other".
Set "sure" to true only if the message clearly matches the label.
Answer only with JSON: {"label": "<label>", "sure": true|false, "reason": "<3 to 6 words>"}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "sure": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["label", "sure", "reason"],
}

# A second, separate call, made only when we are about to ask Noor something:
# one prompt that both classifies and picks among seven questions made the
# 3B model worse at classifying.
QUESTION_PROMPT = """A smallholder coffee farmer on Mount Elgon, Uganda, sent the SMS below (often a rough machine translation from Luganda).
Our system cannot yet tell what is wrong. The possible diagnoses are leaf_rust (orange powder under the leaves), phoma (dark brown, dry or black patches, black dying tips), healthy leaves, or another problem.

Pick the ONE follow-up question whose answer would best settle it. The farmer receives exactly the question text, so it must make sense as a reply to their message:
""" + _question_menu() + """

Never ask about leaves if the message is clearly about berries, stems or roots: pick "clarify" then.
Answer only with JSON: {"question": "<question id>"}."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string", "enum": QUESTIONS}},
    "required": ["question"],
}


@lru_cache(maxsize=1)
def get_client():
    # Imported here, not at module level: without the package llm_predict
    # catches the error and returns label=None, the documented fallback.
    import ollama

    return ollama.Client(timeout=TIMEOUT_S)  # reads OLLAMA_HOST itself


def enabled():
    return os.getenv("USE_LLM", "1") != "0"


def preload() -> None:
    """Load the model in Ollama at startup; a cold first call can exceed TIMEOUT_S."""
    if enabled():
        llm_predict("warm-up")


@lru_cache(maxsize=2048)
def _chat(system: str, user: str, schema_key: str) -> str:
    schema = SCHEMA if schema_key == "label" else QUESTION_SCHEMA
    response = get_client().chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=schema,
        options={"temperature": 0, "num_predict": 60},
        keep_alive="1h",
    )
    return response["message"]["content"]


def _ask(text_en: str) -> str:
    return _chat(SYSTEM_PROMPT, text_en, "label")


def llm_predict(text_en: str) -> dict:
    """-> {"label": str | None, "sure": bool, "reason": str}

    label is None if the LLM is disabled, unreachable or off-schema.
    """
    if not enabled():
        return {"label": None, "sure": False, "reason": "llm_disabled"}
    try:
        result = json.loads(_ask(text_en))
    except Exception as e:  # Ollama down, timeout, invalid JSON...
        return {"label": None, "sure": False, "reason": f"llm_error: {type(e).__name__}"}

    label = result.get("label")
    if label not in LABELS:
        return {"label": None, "sure": False, "reason": f"llm_bad_label: {label!r}"}
    return {"label": label, "sure": bool(result.get("sure")), "reason": str(result.get("reason", ""))[:80]}


def llm_question(text_en: str, guesses: str = "") -> str:
    """The template id of the clarifying question to send. Never fails:
    FALLBACK_QUESTION ("tell me more") if the LLM is off, down or off-schema."""
    if not enabled():
        return FALLBACK_QUESTION
    user = f"SMS: {text_en}" + (f"\n(Guesses so far: {guesses})" if guesses else "")
    try:
        question = json.loads(_chat(QUESTION_PROMPT, user, "question")).get("question")
    except Exception:
        return FALLBACK_QUESTION
    return question if question in QUESTIONS else FALLBACK_QUESTION


if __name__ == "__main__":
    for msg in [
        "orange powder under the leaves",
        "the tips of the young leaves are turning black and drying",
        "my leaves are dark green and shiny",
        "my coffee looks bad",
        "there are brown spots on my leaves",
        "my coffee leaves are turning yellow",
        "the leaves look a bit strange",
        "small holes in the berries",
        "how are you sir",
    ]:
        print(msg, "->", llm_predict(msg), "| ask:", llm_question(msg))
