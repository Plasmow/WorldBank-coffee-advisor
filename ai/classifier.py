"""P1's classifier: e5-small-v2 sentence embedding -> calibrated logistic regression.

Interface used by analyze.py:
    THRESHOLD: float
    classify(text_en: str) -> tuple[str, float]   # (label, calibrated proba)
    preload() -> None

The regression is read from model/classifier.npz (exported from
classifier.joblib by scripts/export_classifier.py), so the server needs numpy
but not scikit-learn, and no pickle crosses sklearn versions. The embedder is
the same one the training embeddings came from (data/embeddings/meta.json):
mean pooling over the last hidden state, then L2 normalisation.
"""

import json
import logging
import os
import pathlib
from functools import lru_cache

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ = ROOT / "model" / "classifier.npz"
LABELS_JSON = ROOT / "model" / "labels.json"


def _threshold():
    try:
        return float(json.loads(LABELS_JSON.read_text(encoding="utf-8"))["threshold"])
    except Exception:
        return 0.8


THRESHOLD = float(os.environ.get("CLASSIFIER_THRESHOLD") or _threshold())


@lru_cache(maxsize=1)
def get_head():
    import numpy as np

    d = np.load(NPZ)
    return {
        "coef": d["coef"],
        "intercept": d["intercept"],
        "classes": [str(c) for c in d["classes"]],
        "temperature": float(d["temperature"]),
        "embedder": str(d["embedder"]),
        "prefix": str(d["prefix"]),
    }


@lru_cache(maxsize=1)
def get_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(max(1, int(os.environ.get("TORCH_THREADS", "2"))))
    name = get_head()["embedder"]
    log.info("loading embedder %s", name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return AutoTokenizer.from_pretrained(name), model


def embed(texts):
    import torch

    tokenizer, model = get_encoder()
    prefix = get_head()["prefix"]
    batch = tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()


def predict_proba(texts):
    """Calibrated probabilities, columns in get_head()["classes"] order."""
    import numpy as np

    head = get_head()
    logits = embed(texts) @ head["coef"].T + head["intercept"]
    z = logits / head["temperature"]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def preload():
    classify("orange powder under the leaves")


@lru_cache(maxsize=2048)
def classify(text_en: str) -> tuple[str, float]:
    p = predict_proba([text_en])[0]
    i = int(p.argmax())
    return get_head()["classes"][i], float(p[i])
