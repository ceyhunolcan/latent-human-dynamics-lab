"""Feature engineering: wearable, behavioral, climate, missingness, baseline.

Every feature function takes a DataFrame in canonical schema and returns a
DataFrame with *additional* columns, leaving the input columns intact.

Conventions
-----------
* All rolling statistics are computed per-participant and respect time
  ordering (the caller is responsible for sorting).
* Deviations from baseline are reported as standardized within-person
  z-scores. The first 7 days of a participant's window have no baseline
  and get NaN, then conservative imputation downstream.
"""

from .wearable_features import compute_wearable_features
from .behavioral_features import compute_behavioral_features
from .climate_features import compute_climate_features
from .missingness_features import compute_missingness_features
from .baseline_features import compute_baseline_features


def engineer_all_features(df):
    """Run all feature modules and return a merged DataFrame.

    Order matters: missingness features are computed *first* on the raw frame
    so they reflect the original dropout pattern. Then we impute safe
    defaults so downstream rolling statistics don't propagate NaNs. The
    order is fixed: missingness → impute → baseline → wearable → behavioral
    → climate.
    """
    # Empty-cohort short-circuit. Calling rolling/groupby on a 0-row
    # DataFrame raises deep inside pandas with a misleading error.
    if len(df) == 0:
        return df.copy()

    out = df.copy()
    out = compute_missingness_features(out)

    # Safe imputation between missingness measurement and downstream rolling
    # statistics. We forward-fill within participant (carry the last observed
    # value) and then back-fill the leading edge. This is conservative —
    # large gaps are still flagged by the missingness features above.
    if "participant_id" in out.columns:
        out = out.sort_values(["participant_id", "date"]).reset_index(drop=True)
        num_cols = out.select_dtypes(include="number").columns
        out[num_cols] = (
            out.groupby("participant_id")[num_cols]
            .transform(lambda g: g.ffill().bfill())
        )
        # Anything still NaN (e.g. a participant whose channel was entirely
        # missing) falls back to the column-wide median.
        for c in num_cols:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())
        # Final fallback for any columns where the median itself is NaN.
        out = out.fillna(0.0)

    out = compute_baseline_features(out)
    out = compute_wearable_features(out)
    out = compute_behavioral_features(out)
    out = compute_climate_features(out)
    return out


__all__ = [
    "compute_wearable_features",
    "compute_behavioral_features",
    "compute_climate_features",
    "compute_missingness_features",
    "compute_baseline_features",
    "engineer_all_features",
]
