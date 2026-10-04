#!/usr/bin/env python3
"""Embed every text of dataset.jsonl with e5-small-v2 (MIT), one file per split.

Input  (data/synthetic/dataset.jsonl): made by make_splits.py from
    leaf_descriptions.jsonl + other_messages.jsonl, using the frozen splits.json.
    Each row already has its split (train / calib / test), so nothing is re-split here.
Output (data/embeddings/):
    train.npz, calib.npz, test.npz   X (n, 384) float32, L2-normalised
                                     label, group, source_id, kind, text  (string arrays, same order as X)
    meta.json                        model, prefix, dimension, rows per split

E5 models expect a prefix: "query: " is the one for classification / symmetric tasks.
Load later with:  d = np.load("data/embeddings/train.npz"); d["X"], d["label"]

Run from the repo root:  python scripts/extract_embeddings.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SPLIT_NAMES = ("train", "calib", "test")
FIELDS = ("label", "group", "source_id", "kind", "text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/synthetic/dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--model", default="intfloat/e5-small-v2")
    ap.add_argument("--prefix", default="query: ", help='E5 prefix; "query: " for classification')
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{args.dataset} is empty. Run scripts/make_splits.py first.")

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model {args.model} ({dim} dims) on {model.device}; {len(rows)} texts")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLIT_NAMES:
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            raise SystemExit(f"No text in split '{split}'")
        X = model.encode([args.prefix + r["text"] for r in sel], batch_size=args.batch_size,
                         normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.savez_compressed(args.out_dir / f"{split}.npz", X=X,
                            **{f: np.array([r[f] for r in sel]) for f in FIELDS})
        counts[split] = dict(Counter(r["label"] for r in sel))
        print(f"  {split:5s} {X.shape}  {counts[split]}")

    (args.out_dir / "meta.json").write_text(json.dumps(
        {"model": args.model, "prefix": args.prefix, "dim": dim, "normalized": True, "rows": counts}, indent=1))
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
