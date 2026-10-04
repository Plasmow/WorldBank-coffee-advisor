#!/usr/bin/env python3
"""Export model/classifier.joblib to model/classifier.npz for the server.

The server computes the logistic regression with numpy, so it needs neither
scikit-learn nor a pickle that only loads with the exact sklearn version that
wrote it. Run again after every train_classifier.py.

    python scripts/export_classifier.py
"""
from pathlib import Path

import joblib
import numpy as np

SRC, OUT = Path("model/classifier.joblib"), Path("model/classifier.npz")

b = joblib.load(SRC)
m = b["model"]
np.savez(
    OUT,
    coef=m.coef_, intercept=m.intercept_, classes=np.array(b["classes"]),
    temperature=b["temperature"], entropy_threshold=b["entropy_threshold"],
    embedder=b["embedder"], prefix=b["prefix"],
)
print(f"{OUT}: classes {b['classes']}, T={b['temperature']:.3f}")
