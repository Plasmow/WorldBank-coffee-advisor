#!/usr/bin/env python3
"""Train the text classifier on the e5-small embeddings and save it to model/classifier.joblib.

Inputs  (data/embeddings/, from extract_embeddings.py): train.npz, calib.npz, test.npz, meta.json

    train : fit a logistic regression (class-balanced); C is picked by calib log-loss
    calib : temperature-scale the probabilities, then choose the "not sure" threshold
    test  : final numbers, printed once at the end. Nothing is tuned on it.

The "not sure" rule: if the highest probability is below the threshold, answer
--reject-label instead. The threshold is the lowest one for which the accepted
calib predictions reach --target-accuracy.

Output  (model/classifier.joblib), a dict:
    model        fitted LogisticRegression
    classes      class names in the order of predict_proba
    temperature  divide the logits by it before the softmax
    threshold    below this top probability -> reject_label
    reject_label label to return when not sure
    embedder, prefix   how to embed a text before calling the model

Use it:
    b = joblib.load("model/classifier.joblib")
    x = SentenceTransformer(b["embedder"]).encode([b["prefix"] + text], normalize_embeddings=True)
    p = softmax(b["model"].decision_function(x) / b["temperature"])

Run from the repo root:  python scripts/train_classifier.py
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss


def load(emb_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(emb_dir / f"{split}.npz")
    return d["X"], d["label"]


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Single scalar T minimising the calib log-loss of softmax(logits / T)."""
    nll = lambda t: -log_softmax(logits / t, axis=1)[np.arange(len(y_idx)), y_idx].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def choose_threshold(proba: np.ndarray, y_idx: np.ndarray, target: float) -> float:
    """Lowest threshold whose accepted predictions reach the target accuracy (else the best accuracy)."""
    top, pred = proba.max(1), proba.argmax(1)
    best_t, best_acc = 1.0, -1.0
    for t in np.unique(np.concatenate([[0.0], np.round(top, 4)])):
        keep = top >= t
        if keep.sum() < 0.3 * len(top):  # do not reject most of the data
            break
        acc = float((pred[keep] == y_idx[keep]).mean())
        if acc >= target:
            return float(t)
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def report(name: str, proba, y_idx, classes, threshold, reject_label) -> None:
    top, pred = proba.max(1), proba.argmax(1)
    accepted = top >= threshold
    print(f"\n== {name} ({len(y_idx)} texts) ==")
    print(f"accuracy without rejection : {(pred == y_idx).mean():.3f}")
    print(f"answered (>= {threshold:.3f})      : {accepted.mean():.1%}, accuracy on them {(pred[accepted] == y_idx[accepted]).mean():.3f}")
    print(f"rejected as '{reject_label}'     : {(~accepted).mean():.1%}")
    print(classification_report(y_idx, pred, labels=range(len(classes)), target_names=classes, digits=3, zero_division=0))
    print("confusion matrix (rows = true, columns = predicted):\n", confusion_matrix(y_idx, pred, labels=range(len(classes))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--out", type=Path, default=Path("model/classifier.joblib"))
    ap.add_argument("--c-grid", type=float, nargs="+", default=[1, 3, 10, 30, 100, 300, 1000])
    ap.add_argument("--target-accuracy", type=float, default=0.99, help="accuracy wanted on answered messages")
    ap.add_argument("--reject-label", default="unknown")
    args = ap.parse_args()

    meta = json.loads((args.emb_dir / "meta.json").read_text())
    (Xtr, ytr), (Xca, yca), (Xte, yte) = (load(args.emb_dir, s) for s in ("train", "calib", "test"))
    classes = sorted(str(c) for c in set(ytr))
    idx = {c: i for i, c in enumerate(classes)}
    ytr_i, yca_i, yte_i = (np.array([idx[c] for c in y]) for y in (ytr, yca, yte))
    print(f"classes: {classes}; train {len(ytr)}, calib {len(yca)}, test {len(yte)}")

    # C chosen on calib log-loss
    fits = {}
    for c in args.c_grid:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(Xtr, ytr_i)
        fits[c] = (m, log_loss(yca_i, m.predict_proba(Xca), labels=range(len(classes))))
        print(f"  C={c:<5g} calib log-loss {fits[c][1]:.4f}")
    best_c = min(fits, key=lambda c: fits[c][1])
    model = fits[best_c][0]
    print(f"-> C = {best_c}")

    temperature = fit_temperature(model.decision_function(Xca), yca_i)
    proba = lambda X: softmax(model.decision_function(X) / temperature, axis=1)
    threshold = choose_threshold(proba(Xca), yca_i, args.target_accuracy)
    print(f"temperature = {temperature:.3f}, threshold = {threshold:.3f}")

    report("calib (used to set the threshold)", proba(Xca), yca_i, classes, threshold, args.reject_label)
    report("TEST (final)", proba(Xte), yte_i, classes, threshold, args.reject_label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": classes, "temperature": temperature, "threshold": threshold,
                 "reject_label": args.reject_label, "embedder": meta["model"], "prefix": meta["prefix"]}, args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
