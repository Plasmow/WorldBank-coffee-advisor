#!/usr/bin/env python3
"""Evaluate model/classifier.joblib with three separate numbers.

For every message the model either ANSWERS with a class or ABSTAINS ("unknown", when the
entropy is above the threshold). Each message therefore ends in one of three outcomes:

    correct          answered, and right
    confident error  answered, and wrong      <- the dangerous one: nobody knows it is wrong
    abstention       "unknown"                <- the safe fallback: a human can look at it

and  correct + confident errors + abstentions = all messages. Reported:

    confident-error rate = confident errors / all messages
    abstention rate      = abstentions / all messages
    accuracy on answered = correct / (correct + confident errors)

Accuracy on answered alone hides the abstentions (abstain a lot and it looks great);
the abstention rate alone hides the errors. Look at the three together.
An abstention on a message whose true label is "other" is harmless, so it is also
reported separately from abstentions on real disease / healthy messages.

Data (one or more of):
    --split test|calib   texts of data/embeddings/<split>.npz (already embedded)
    --csv FILE           a csv with columns text,label (embedded here with the model's embedder),
                         e.g. data/eval/hard_test.csv
With no option: --split test and --csv data/eval/hard_test.csv (if it exists).

Run from the repo root:  python scripts/evaluate.py [--show] [--threshold 0.4]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from train_classifier import entropy_norm

OTHER = "other"


def predict(bundle: dict, X: np.ndarray, threshold: float | None = None) -> tuple[list[str], np.ndarray]:
    """(class or reject label for every row, normalised entropy)."""
    p = softmax(bundle["model"].decision_function(X) / bundle["temperature"], axis=1)
    h = entropy_norm(p)
    thr = bundle["entropy_threshold"] if threshold is None else threshold
    out = [bundle["reject_label"] if hh > thr else bundle["classes"][i] for i, hh in zip(p.argmax(1), h)]
    return out, h


def load_split(emb_dir: Path, split: str) -> tuple[np.ndarray, list[str], list[str]]:
    d = np.load(emb_dir / f"{split}.npz")
    return d["X"], [str(t) for t in d["text"]], [str(l) for l in d["label"]]


def load_csv(path: Path, bundle: dict) -> tuple[np.ndarray, list[str], list[str]]:
    from sentence_transformers import SentenceTransformer  # only needed for csv files

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
    X = SentenceTransformer(bundle["embedder"]).encode(
        [bundle["prefix"] + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
    return X, texts, labels


def evaluate(name: str, bundle: dict, X, texts, labels, threshold, show: bool) -> None:
    reject = bundle["reject_label"]
    pred, h = predict(bundle, X, threshold)
    n = len(labels)
    abstain = [p == reject for p in pred]
    correct = [p == l for p, l in zip(pred, labels)]
    confident_err = [(not a) and (not c) for a, c in zip(abstain, correct)]
    n_ans = n - sum(abstain)
    n_abs, n_err, n_ok = sum(abstain), sum(confident_err), sum(correct)
    abs_other = sum(a and l == OTHER for a, l in zip(abstain, labels))
    missed = sum(e and p == OTHER and l != OTHER for e, p, l in zip(confident_err, pred, labels))

    print(f"\n=== {name}: {n} messages (entropy threshold {threshold if threshold is not None else bundle['entropy_threshold']:.3f}) ===")
    print(f"  correct answers       : {n_ok:4d}  {n_ok / n:6.1%}")
    print(f"  confident errors      : {n_err:4d}  {n_err / n:6.1%}   <- wrong and not flagged")
    print(f"  abstentions (unknown) : {n_abs:4d}  {n_abs / n:6.1%}   "
          f"({abs_other} on true '{OTHER}' = harmless, {n_abs - abs_other} on in-scope messages = lost answers)")
    print(f"  accuracy on answered  : {n_ok / n_ans:6.1%}   ({n_ok}/{n_ans} answered)" if n_ans else "  nothing answered")
    print(f"  of the confident errors, in-scope message labelled '{OTHER}' (missed problem): {missed}")

    per = Counter()
    for l, a, c in zip(labels, abstain, correct):
        per[(l, "abstain" if a else "correct" if c else "error")] += 1
    print(f"  {'true label':12s}{'n':>5s}{'correct':>9s}{'error':>7s}{'abstain':>9s}")
    for l in sorted(set(labels)):
        k = sum(per[(l, o)] for o in ("correct", "error", "abstain"))
        print(f"  {l:12s}{k:5d}{per[(l, 'correct')]:9d}{per[(l, 'error')]:7d}{per[(l, 'abstain')]:9d}")

    if show:
        print("  confident errors:")
        for t, l, p, hh, e in zip(texts, labels, pred, h, confident_err):
            if e:
                print(f"    {l} -> {p} (H={hh:.2f})  {t[:80]}")
        print("  abstentions on in-scope messages:")
        for t, l, a, hh in zip(texts, labels, abstain, h):
            if a and l != OTHER:
                print(f"    {l} (H={hh:.2f})  {t[:80]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("model/classifier.joblib"))
    ap.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--split", action="append", choices=("train", "calib", "test"), default=[])
    ap.add_argument("--csv", action="append", type=Path, default=[])
    ap.add_argument("--threshold", type=float, default=None, help="override the saved entropy threshold")
    ap.add_argument("--show", action="store_true", help="list the confident errors and the lost answers")
    args = ap.parse_args()

    if not args.split and not args.csv:
        args.split = ["test"]
        default_csv = Path("data/eval/hard_test.csv")
        args.csv = [default_csv] if default_csv.exists() else []

    bundle = joblib.load(args.model)
    for s in args.split:
        evaluate(f"split '{s}'", bundle, *load_split(args.emb_dir, s), args.threshold, args.show)
    for c in args.csv:
        evaluate(c.name, bundle, *load_csv(c, bundle), args.threshold, args.show)


if __name__ == "__main__":
    main()
