#!/usr/bin/env python
"""Download every model the server needs, once, at build time.

    python scripts/fetch_models.py

- the Opus-MT lg->en fine-tune in CTranslate2 int8 (~80 MB) from
  TRANSLATOR_REPO into model/opus-lg-en-ct2/, tokenizer included;
- e5-small-v2 into the Hugging Face cache (HF_HOME), for the classifier.

On Render this runs in the build command, so a deploy never downloads at
boot and a cold start only loads from disk. Weights never go in git.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from ai import translate

    path = translate.model_dir()
    if (path / "model.bin").exists():
        print(f"translator: {path} already there")
    else:
        # _ensure_model does the download, and knows where it goes.
        print(f"translator: fetching into {path}")
        translate._ensure_model()

    # The tokenizer ships inside the model folder, so nothing else to fetch
    # for translation. The classifier's embedder is a separate download.
    npz = ROOT / "model" / "classifier.npz"
    if not npz.exists():
        print("classifier: model/classifier.npz missing, skipping the embedder")
        return
    try:
        import numpy as np
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"classifier: {exc}; install requirements-ai.txt to fetch the embedder")
        return

    embedder = str(np.load(npz)["embedder"])
    print(f"embedder: {embedder}")
    AutoTokenizer.from_pretrained(embedder)
    AutoModel.from_pretrained(embedder)
    print("all models ready")


if __name__ == "__main__":
    main()
