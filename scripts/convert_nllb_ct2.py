#!/usr/bin/env python
"""Convert NLLB-200-distilled-600M to CTranslate2 int8. Run once, on a laptop.

    uv pip install -r requirements-convert.txt
    uv run python scripts/convert_nllb_ct2.py

Downloads ~2.4 GB and writes ~600 MB into model/nllb-ct2-int8/. That folder is
gitignored: weights never go in git. To get it onto a host, either upload the
folder or push it to a Hugging Face repo and set NLLB_CT2_REPO.

torch reads the Hugging Face weights here; the translator itself runs on CTranslate2 alone.
"""

import pathlib
import subprocess
import sys

MODEL = "facebook/nllb-200-distilled-600M"
OUT = pathlib.Path("model/nllb-ct2-int8")
# Copied next to the weights so the tokenizer can never drift from the model.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def main():
    if (OUT / "model.bin").exists():
        print(f"{OUT} already converted. Delete it to redo the conversion.")
        return 0

    cmd = [
        "ct2-transformers-converter",
        "--model", MODEL,
        "--output_dir", str(OUT),
        "--quantization", "int8",
        "--low_cpu_mem_usage",
        "--copy_files", *TOKENIZER_FILES,
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode:
        return result.returncode

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n{OUT}  {size / 1e6:.0f} MB")
    print("Check it with:  uv run python -c \""
          "from ai.translate import lug_to_en, load; load(); print(lug_to_en('Ebikoola birimu obulwadde'))\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
