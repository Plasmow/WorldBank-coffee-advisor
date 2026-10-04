#!/usr/bin/env python3
"""Evaluate model/classifier.joblib: three separate numbers, a threshold sweep and a confusion matrix.

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

An abstention is a message handed over to a human (the "handoff rate"). Also printed:
    - a threshold sweep: for several entropy thresholds, the precision (accuracy on answered),
      the handoff rate and the confident-error rate. It shows what a stricter or looser threshold
      would cost; the line marked "<- saved" is the threshold stored in the model;
    - a confusion matrix (rows = true label, columns = predicted label, last column = handed over).

Data (one or more of):
    --split test|calib   texts of data/embeddings/<split>.npz (already embedded)
    --csv FILE           a csv with columns text,label (embedded here with the model's embedder),
                         e.g. data/eval/hard_test.csv
With no option: --split test and --csv data/eval/hard_test.csv (if it exists).

Robustness to translation (--roundtrip): every text is translated English -> Luganda -> English
with NLLB (ai/translate.py; convert the weights once with scripts/convert_nllb_ct2.py), re-embedded
and evaluated again. This mimics a farmer writing in Luganda whose message is translated before it
reaches the classifier. The report compares it with the original English texts: change in correct
answers / confident errors / handoffs, and how many predictions changed.

Run from the repo root:  python scripts/evaluate.py [--show] [--threshold 0.4]
"""
import argparse
import csv
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from train_classifier import entropy_norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for ai.translate

OTHER = "other"


def predict(bundle: dict, X: np.ndarray, threshold: float | None = None) -> tuple[list[str], np.ndarray, list[str]]:
    """(class or reject label for every row, normalised entropy, most likely class before any rejection)."""
    p = softmax(bundle["model"].decision_function(X) / bundle["temperature"], axis=1)
    h = entropy_norm(p)
    thr = bundle["entropy_threshold"] if threshold is None else threshold
    top = [bundle["classes"][i] for i in p.argmax(1)]
    out = [bundle["reject_label"] if hh > thr else c for c, hh in zip(top, h)]
    return out, h, top


def load_split(emb_dir: Path, split: str) -> tuple[np.ndarray, list[str], list[str]]:
    d = np.load(emb_dir / f"{split}.npz")
    return d["X"], [str(t) for t in d["text"]], [str(l) for l in d["label"]]


@lru_cache(maxsize=1)
def _embedder(name: str):
    from sentence_transformers import SentenceTransformer  # only needed for csv files / round trip

    return SentenceTransformer(name)


def embed(bundle: dict, texts: list[str]) -> np.ndarray:
    return _embedder(bundle["embedder"]).encode(
        [bundle["prefix"] + t for t in texts], normalize_embeddings=True, show_progress_bar=False)


def load_csv(path: Path, bundle: dict) -> tuple[np.ndarray, list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
    return embed(bundle, texts), texts, labels


def round_trip(texts: list[str]) -> list[str]:
    """English -> Luganda -> English with NLLB."""
    from ai import translate

    translate.load()
    return [translate.lug_to_en(translate.en_to_lug(t)) for t in texts]


SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def sweep_table(h: np.ndarray, top: list[str], labels: list[str], saved: float) -> None:
    """Precision / handoff rate / confident-error rate for several entropy thresholds."""
    right = np.array([c == l for c, l in zip(top, labels)])
    print("  threshold sweep (precision = accuracy on answered; handoff = sent to a human):")
    print(f"  {'entropy <=':>11s}{'precision':>11s}{'handoff':>9s}{'conf. error':>13s}")
    for t in sorted(set(SWEEP) | {round(saved, 3)}):
        answered = h <= t
        k = int(answered.sum())
        ok = int((right & answered).sum())
        prec = f"{ok / k:11.1%}" if k else f"{'-':>11s}"
        mark = "  <- saved" if abs(t - round(saved, 3)) < 1e-9 else ""
        print(f"  {t:11.3f}{prec}{1 - k / len(h):9.1%}{(k - ok) / len(h):13.1%}{mark}")


def confusion(labels: list[str], pred: list[str], classes: list[str], reject: str) -> None:
    """Rows = true label, columns = predicted label; the last column is the messages handed over."""
    cols = list(classes) + [reject]
    m = Counter(zip(labels, pred))
    rows = [c for c in classes if c in set(labels)]
    print("  confusion matrix (rows = true, columns = predicted):")
    print(f"  {'':12s}" + "".join(f"{c[:10]:>11s}" for c in cols))
    for r in rows:
        print(f"  {r:12s}" + "".join(f"{m[(r, c)]:11d}" for c in cols))


def evaluate(name: str, bundle: dict, X, texts, labels, threshold, show: bool, detail: bool = True) -> dict:
    reject = bundle["reject_label"]
    pred, h, top = predict(bundle, X, threshold)
    saved = bundle["entropy_threshold"] if threshold is None else threshold
    n = len(labels)
    abstain = [p == reject for p in pred]
    correct = [p == l for p, l in zip(pred, labels)]
    confident_err = [(not a) and (not c) for a, c in zip(abstain, correct)]
    n_ans = n - sum(abstain)
    n_abs, n_err, n_ok = sum(abstain), sum(confident_err), sum(correct)
    abs_other = sum(a and l == OTHER for a, l in zip(abstain, labels))
    missed = sum(e and p == OTHER and l != OTHER for e, p, l in zip(confident_err, pred, labels))

    print(f"\n=== {name}: {n} messages (entropy threshold {saved:.3f}) ===")
    print(f"  correct answers       : {n_ok:4d}  {n_ok / n:6.1%}")
    print(f"  confident errors      : {n_err:4d}  {n_err / n:6.1%}   <- wrong and not flagged")
    print(f"  handoff (unknown)     : {n_abs:4d}  {n_abs / n:6.1%}   "
          f"({abs_other} on true '{OTHER}' = harmless, {n_abs - abs_other} on in-scope messages = lost answers)")
    print(f"  accuracy on answered  : {n_ok / n_ans:6.1%}   ({n_ok}/{n_ans} answered)" if n_ans else "  nothing answered")
    print(f"  of the confident errors, in-scope message labelled '{OTHER}' (missed problem): {missed}")

    summary = {"pred": pred, "correct": n_ok, "error": n_err, "handoff": n_abs, "n": n}
    if not detail:
        return summary

    print()
    sweep_table(h, top, labels, saved)
    print()
    confusion(labels, pred, bundle["classes"], reject)
    print()
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
    return summary


def compare(name: str, before: dict, after: dict, texts, translated, labels, show: bool) -> None:
    """Original English vs the same messages after the Luganda round trip."""
    n = before["n"]
    changed = [i for i, (a, b) in enumerate(zip(before["pred"], after["pred"])) if a != b]
    print(f"\n--- {name}: original vs after NLLB round trip (en -> lug -> en), {n} messages ---")
    print(f"  {'':22s}{'original':>10s}{'translated':>12s}{'change':>9s}")
    for key, label in (("correct", "correct answers"), ("error", "confident errors"), ("handoff", "handoff (unknown)")):
        a, b = before[key] / n, after[key] / n
        print(f"  {label:22s}{a:10.1%}{b:12.1%}{(b - a) * 100:+8.1f}pt")
    print(f"  predictions that changed: {len(changed)}/{n} ({len(changed) / n:.1%})")
    if show:
        for i in changed:
            print(f"    [{labels[i]}] {before['pred'][i]} -> {after['pred'][i]}")
            print(f"       en : {texts[i][:90]}")
            print(f"       rt : {translated[i][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("model/classifier.joblib"))
    ap.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--split", action="append", choices=("train", "calib", "test"), default=[])
    ap.add_argument("--csv", action="append", type=Path, default=[])
    ap.add_argument("--threshold", type=float, default=None, help="override the saved entropy threshold")
    ap.add_argument("--show", action="store_true", help="list the confident errors and the lost answers")
    ap.add_argument("--roundtrip", action="store_true",
                    help="also evaluate after an English -> Luganda -> English NLLB round trip")
    ap.add_argument("--max-roundtrip", type=int, default=300,
                    help="translate at most this many texts per dataset (first ones, in order)")
    args = ap.parse_args()

    if not args.split and not args.csv:
        args.split = ["test"]
        default_csv = Path("data/eval/hard_test.csv")
        args.csv = [default_csv] if default_csv.exists() else []

    bundle = joblib.load(args.model)
    datasets = [(f"split '{s}'", load_split(args.emb_dir, s)) for s in args.split]
    datasets += [(c.name, load_csv(c, bundle)) for c in args.csv]
    for name, (X, texts, labels) in datasets:
        before = evaluate(name, bundle, X, texts, labels, args.threshold, args.show)
        if args.roundtrip:
            k = min(args.max_roundtrip, len(texts))
            print(f"\nTranslating {k} texts of {name} (en -> lug -> en)...")
            translated = round_trip(texts[:k])
            sub = evaluate(name + " (round trip)", bundle, embed(bundle, translated), translated, labels[:k],
                           args.threshold, False, detail=False)
            ref = evaluate(name + " (same subset, original)", bundle, X[:k], texts[:k], labels[:k],
                           args.threshold, False, detail=False)
            compare(name, ref, sub, texts[:k], translated, labels[:k], args.show)


if __name__ == "__main__":
    main()
