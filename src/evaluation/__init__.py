"""Evaluation suite.

The evaluation modules cover six areas: prediction metrics, calibration,
robustness under distribution shift, subgroup analysis, representation
analysis of the latent space, and synthetic-to-real distribution similarity.

Functions return plain dicts and dataframes wherever possible so the results
can be serialised, written to disk, and surfaced in the dashboard.
"""

from .metrics import (
    regression_metrics,
    classification_metrics,
    trajectory_rmse,
)
from .calibration import calibration_curve_data, expected_calibration_error
from .robustness import (
    missingness_stress_test,
    heatwave_subgroup_analysis,
    climate_vulnerability_subgroup_analysis,
    out_of_distribution_environmental_shock_test,
)
from .subgroup_analysis import subgroup_performance_table
from .representation_analysis import (
    representation_pca,
    cluster_separation_score,
    latent_to_ground_truth_correlation,
    trajectory_stability,
)
from .synthetic_to_real import (
    distribution_similarity,
    correlation_matrix_similarity,
    missingness_pattern_comparison,
    synthetic_to_real_report,
)

__all__ = [
    "regression_metrics",
    "classification_metrics",
    "trajectory_rmse",
    "calibration_curve_data",
    "expected_calibration_error",
    "missingness_stress_test",
    "heatwave_subgroup_analysis",
    "climate_vulnerability_subgroup_analysis",
    "out_of_distribution_environmental_shock_test",
    "subgroup_performance_table",
    "representation_pca",
    "cluster_separation_score",
    "latent_to_ground_truth_correlation",
    "trajectory_stability",
    "distribution_similarity",
    "correlation_matrix_similarity",
    "missingness_pattern_comparison",
    "synthetic_to_real_report",
]
