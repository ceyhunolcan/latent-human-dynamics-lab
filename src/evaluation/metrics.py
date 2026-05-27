"""Prediction metrics: regression, classification, and trajectory-level RMSE."""

from __future__ import annotations

from typing import Optional

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, and explained variance for a regression task."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> dict:
    """Accuracy, macro F1, plus optional macro AUROC/AUPRC when proba given."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        classes = list(np.unique(y_true))
        try:
            yb = label_binarize(y_true, classes=classes)
            if yb.shape[1] == 1:
                yb = np.hstack([1 - yb, yb])
            out["auroc_ovr_macro"] = float(
                roc_auc_score(yb, y_proba, average="macro", multi_class="ovr")
            )
            out["auprc_macro"] = float(average_precision_score(yb, y_proba, average="macro"))
        except Exception:
            out["auroc_ovr_macro"] = float("nan")
            out["auprc_macro"] = float("nan")
    return out


def trajectory_rmse(true_traj: np.ndarray, pred_traj: np.ndarray) -> float:
    """RMSE over all time steps and dimensions of two latent trajectories."""
    a = np.asarray(true_traj, dtype=float)
    b = np.asarray(pred_traj, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))
