"""Downstream prediction tasks operating on encoded latent states.

These are deliberately simple baselines: a logistic regression for regime
classification, a random-forest reference for the same task, and a small
container class that the API can use to attach pretrained heads.

The point of these baselines is comparison, not state of the art. They make
the leaderboard interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DownstreamTaskHead:
    """Container for a fitted scikit-learn estimator and its scaler."""

    model: object
    scaler: object
    class_labels: Optional[list] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict(Xs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)
        return None  # type: ignore


def train_logistic_baseline(X: np.ndarray, y: np.ndarray) -> DownstreamTaskHead:
    """Fit a standardised logistic regression on (X, y)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=500).fit(Xs, y)
    return DownstreamTaskHead(model=model, scaler=scaler, class_labels=list(np.unique(y)))


def evaluate_baseline(head: DownstreamTaskHead, X: np.ndarray, y: np.ndarray) -> dict:
    """Return AUROC (macro/OvR), AUPRC, F1, accuracy for a fitted head."""
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        accuracy_score,
    )
    from sklearn.preprocessing import label_binarize

    y_pred = head.predict(X)
    proba = head.predict_proba(X)
    classes = head.class_labels or list(np.unique(y))

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    auroc = float("nan")
    auprc = float("nan")
    if proba is not None and len(classes) > 1:
        try:
            yb = label_binarize(y, classes=classes)
            if yb.shape[1] == 1:
                yb = np.hstack([1 - yb, yb])
            auroc = roc_auc_score(yb, proba, average="macro", multi_class="ovr")
            auprc = average_precision_score(yb, proba, average="macro")
        except Exception:
            pass

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auroc_ovr_macro": float(auroc),
        "auprc_macro": float(auprc),
    }
