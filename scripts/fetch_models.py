#!/usr/bin/env python
"""Download every model the server needs, once, at build time.

    python scripts/fetch_models.py

- NLLB-200-distilled-600M in CTranslate2 int8 (~620 MB) from NLLB_CT2_REPO
  into model/nllb-ct2-int8/ (skipped if a local copy already exists);
- the NLLB tokenizer and e5-small-v2 into the Hugging Face cache (HF_HOME).

On Render this runs in the build command, so a deploy never downloads at
boot and a cold start only loads from disk. Weights never go in git.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"


def main():
    from huggingface_hub import snapshot_download

    from ai import translate

    path = translate.model_dir()
    if (path / "model.bin").exists():
        print(f"NLLB: {path} already there")
    else:
        repo = os.environ.get("NLLB_CT2_REPO") or DEFAULT_REPO
        target = pathlib.Path(os.environ.get("NLLB_CT2_DIR") or translate.DEFAULT_DIRS[0])
        print(f"NLLB: downloading {repo} into {target}")
        snapshot_download(repo, local_dir=str(target),
                          allow_patterns=["model.bin", "config.json", "shared_vocabulary.*"])

    from transformers import AutoModel, AutoTokenizer

    print(f"tokenizer: {translate.TOKENIZER_NAME}")
    AutoTokenizer.from_pretrained(translate.TOKENIZER_NAME)

    import numpy as np

    embedder = str(np.load(ROOT / "model" / "classifier.npz")["embedder"])
    print(f"embedder: {embedder}")
    AutoTokenizer.from_pretrained(embedder)
    AutoModel.from_pretrained(embedder)
    print("all models ready")


if __name__ == "__main__":
    main()
