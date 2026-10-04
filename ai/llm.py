"""Second opinion: qwen2.5:3b via Ollama, JSON schema enforced, temperature 0.

The LLM never writes text for Noor. It picks a label among the 4 classes,
which analyze() compares with the classifier's, and, for when the two are in
doubt, the id of the one clarifying question (a key of data/templates.json)
that would best settle it.

OLLAMA_HOST points the client at any Ollama server (the laptop through a
tunnel, for a cloud host that cannot run one). Unreachable -> label None, and
analyze() falls back to the calibrated classifier alone (plan B of the roadmap).
"""
import json
import os
from functools import lru_cache

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", "30"))

LABELS = ["leaf_rust", "phoma", "healthy", "other"]
# Template ids of the clarifying questions the LLM may choose from.
QUESTIONS = ["clarify", "clarify_where", "clarify_colour", "clarify_rust_phoma"]

SYSTEM_PROMPT = """You classify short SMS messages sent by smallholder arabica coffee farmers on Mount Elgon, Uganda.
The text is often a rough machine translation from Luganda. The farmer describes what they see on their coffee plants.

Pick exactly one label:
- leaf_rust: orange or yellow powdery/dusty spots, usually on the underside of leaves; pale yellow patches on the upper side; leaves falling early.
- phoma: dark brown, reddish-brown or black patches or spots on leaves, often at tips or edges, dry and papery; young leaves and shoot tips turning black and dying back, often in cold, wet weather.
- healthy: leaves described as green, glossy, uniform, without spots, patches or powder.
- other: anything else - insects, holes, problems on berries, stems or roots, wilting of the whole tree, prices, greetings, unrelated or too vague messages.

Then pick the one question to ask the farmer if we are still unsure:
- clarify_where: the message does not say which part of the plant is affected (e.g. "my coffee looks bad").
- clarify_colour: the message is about leaves but does not say what the marks look like.
- clarify_rust_phoma: the message describes spots or patches on leaves that could be either leaf_rust or phoma.
- clarify: anything else.

Set "sure" to true only if the message clearly matches one label.
Answer only with JSON: {"label": "<label>", "sure": true|false, "question": "<question id>", "reason": "<3 to 6 words>"}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "sure": {"type": "boolean"},
        "question": {"type": "string", "enum": QUESTIONS},
        "reason": {"type": "string"},
    },
    "required": ["label", "sure", "question", "reason"],
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


@lru_cache(maxsize=1024)
def _ask(text_en: str) -> str:
    response = get_client().chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_en},
        ],
        format=SCHEMA,
        options={"temperature": 0, "num_predict": 60},
        keep_alive="1h",
    )
    return response["message"]["content"]


def llm_predict(text_en: str) -> dict:
    """-> {"label": str | None, "sure": bool, "question": str, "reason": str}

    label is None if the LLM is disabled, unreachable or off-schema.
    """
    if not enabled():
        return {"label": None, "sure": False, "question": "clarify", "reason": "llm_disabled"}
    try:
        result = json.loads(_ask(text_en))
    except Exception as e:  # Ollama down, timeout, invalid JSON...
        return {"label": None, "sure": False, "question": "clarify",
                "reason": f"llm_error: {type(e).__name__}"}

    label = result.get("label")
    question = result.get("question")
    if question not in QUESTIONS:
        question = "clarify"
    if label not in LABELS:
        return {"label": None, "sure": False, "question": question,
                "reason": f"llm_bad_label: {label!r}"}
    return {"label": label, "sure": bool(result.get("sure")), "question": question,
            "reason": str(result.get("reason", ""))[:80]}


if __name__ == "__main__":
    for msg in [
        "orange powder under the leaves",
        "the tips of the young leaves are turning black and drying",
        "my leaves are dark green and shiny",
        "my coffee looks bad",
        "what is the price of coffee today?",
    ]:
        print(msg, "->", llm_predict(msg))
