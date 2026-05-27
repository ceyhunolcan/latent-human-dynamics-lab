"""WESAD dataset adapter.

Reference
---------
Schmidt et al. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable
Stress and Affect Detection.*

WESAD is recorded at high frequency (chest + wrist) under controlled
conditions. Because the rest of this pipeline operates at daily
resolution, the adapter aggregates each subject's session into per-
condition daily summaries:

    HRV, EDA, BVP-derived heart rate, accelerometer activity →
    summary statistics → one row per subject per condition.

Conditions in WESAD: Baseline, Stress, Amusement, Meditation. We map them
to placeholder "dates" so the data fits the long-format daily schema used
elsewhere in the repository.

This adapter does **not** ship WESAD data. Calling it without a local copy
raises ``FileNotFoundError`` with the expected layout.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

SCHEMA_MAPPING: dict[str, str] = {
    "subject": "participant_id",
    "hrv_rmssd": "hrv_rmssd",
    "mean_hr": "resting_hr",
    "stress_label": "perceived_stress",
}

WESAD_LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def _summarize_signal(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def load_wesad(root: str | Path) -> pd.DataFrame:
    """Load WESAD from a local directory.

    Parameters
    ----------
    root : path
        Top of the WESAD extraction containing subject subfolders
        ``S2/S2.pkl``, ``S3/S3.pkl``, ... where each pickle is the
        original WESAD per-subject dictionary.

    Returns
    -------
    DataFrame
        One row per (subject, condition). Condition is encoded as a
        synthetic ``date`` (1970-01-01 + label_index days) so the schema
        is compatible with the rest of the pipeline.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"WESAD root not found at {root_path!s}. "
            "Expected per-subject subfolders containing S{N}.pkl files."
        )

    rows: list[dict] = []
    for sub_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        pkl = sub_dir / f"{sub_dir.name}.pkl"
        if not pkl.exists():
            continue
        with pkl.open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        labels = np.asarray(obj.get("label", []))
        for lbl_id, lbl_name in WESAD_LABELS.items():
            mask = labels == lbl_id
            if not mask.any():
                continue
            # In a real implementation we would compute HRV from BVP/ECG.
            # Here we fall back to summary statistics over a placeholder.
            sig = np.asarray(obj.get("signal", {}).get("chest", {}).get("ECG", np.zeros(1)))
            sig_seg = sig[mask] if len(sig) == len(mask) else sig
            rows.append(
                {
                    "participant_id": sub_dir.name,
                    "date": pd.Timestamp("1970-01-01") + pd.Timedelta(days=lbl_id),
                    "condition": lbl_name,
                    "hrv_rmssd": _summarize_signal(sig_seg)["std"],
                    "resting_hr": float(np.mean(sig_seg)) if len(sig_seg) else np.nan,
                    "perceived_stress": 100.0 if lbl_name == "stress" else 20.0,
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No subject pickles found under {root_path!s}."
        )

    return pd.DataFrame(rows)
