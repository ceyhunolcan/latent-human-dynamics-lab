"""Synthetic-to-real validation.

Compares the marginal distributions, joint correlation structure, and
missingness patterns of two cohorts: the synthetic cohort produced by
`src/data/synthetic_generator.py` and a real (or real-like) cohort loaded
via the adapters. Returns a structured report with numeric similarity
scores and matplotlib figures.

These diagnostics are necessary but not sufficient for transferability:
they will catch obvious distributional drift, but cannot detect subtle
mechanistic divergences that would only manifest under perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def distribution_similarity(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Per-column Kolmogorov–Smirnov and Wasserstein-1 distance.

    Smaller values indicate closer distributions. Columns absent from one
    side are skipped.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    if columns is None:
        columns = [c for c in synthetic.columns if c in real.columns]

    rows = []
    for c in columns:
        s = pd.to_numeric(synthetic[c], errors="coerce").dropna().to_numpy()
        r = pd.to_numeric(real[c], errors="coerce").dropna().to_numpy()
        if len(s) < 10 or len(r) < 10:
            continue
        ks_stat, ks_p = ks_2samp(s, r)
        wd = wasserstein_distance(s, r)
        rows.append(
            {
                "column": c,
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "wasserstein_distance": float(wd),
                "synthetic_mean": float(np.mean(s)),
                "real_mean": float(np.mean(r)),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix_similarity(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: Optional[Iterable] = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Compare correlation matrices via Frobenius distance.

    Returns (frobenius_distance, synthetic_corr, real_corr). All NaNs in
    the correlation matrices are zero-filled before the distance is taken.
    """
    if columns is None:
        columns = [
            c
            for c in synthetic.columns
            if c in real.columns and pd.api.types.is_numeric_dtype(synthetic[c])
        ]
    sc = synthetic[list(columns)].corr().fillna(0.0)
    rc = real[list(columns)].corr().fillna(0.0)
    common = sc.index.intersection(rc.index)
    sc = sc.loc[common, common]
    rc = rc.loc[common, common]
    frob = float(np.linalg.norm(sc.to_numpy() - rc.to_numpy(), ord="fro"))
    return frob, sc, rc


def missingness_pattern_comparison(synthetic: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Compare per-column missingness rates between cohorts.

    Returns an empty (but correctly-typed) DataFrame when the two cohorts
    share no columns.
    """
    cols = [c for c in synthetic.columns if c in real.columns]
    rows = []
    for c in cols:
        rows.append(
            {
                "column": c,
                "synthetic_missing_rate": float(synthetic[c].isna().mean()),
                "real_missing_rate": float(real[c].isna().mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["column", "synthetic_missing_rate", "real_missing_rate", "absolute_gap"]
        )
    df = pd.DataFrame(rows)
    df["absolute_gap"] = (df["synthetic_missing_rate"] - df["real_missing_rate"]).abs()
    return df.sort_values("absolute_gap", ascending=False)


@dataclass
class SyntheticToRealReport:
    distribution_table: pd.DataFrame
    correlation_frobenius: float
    correlation_synthetic: pd.DataFrame
    correlation_real: pd.DataFrame
    missingness_table: pd.DataFrame

    def as_markdown(self) -> str:
        lines = ["# Synthetic-to-real validation report\n"]
        lines.append("## Distribution similarity (top 10 most divergent columns)\n")
        if not self.distribution_table.empty:
            top = self.distribution_table.sort_values("ks_statistic", ascending=False).head(10)
            lines.append(top.to_markdown(index=False))
        else:
            lines.append("_no columns evaluated_")
        lines.append(
            f"\n\n## Correlation matrix Frobenius distance: {self.correlation_frobenius:.4f}\n"
        )
        lines.append("## Missingness pattern (top 10 most divergent columns)\n")
        if not self.missingness_table.empty:
            lines.append(self.missingness_table.head(10).to_markdown(index=False))
        return "\n".join(lines)


def synthetic_to_real_report(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: Optional[Iterable] = None,
) -> SyntheticToRealReport:
    """Run all three comparisons and bundle the result."""
    dist_table = distribution_similarity(synthetic, real, columns)
    frob, sc, rc = correlation_matrix_similarity(synthetic, real, columns)
    miss_table = missingness_pattern_comparison(synthetic, real)
    return SyntheticToRealReport(
        distribution_table=dist_table,
        correlation_frobenius=frob,
        correlation_synthetic=sc,
        correlation_real=rc,
        missingness_table=miss_table,
    )
