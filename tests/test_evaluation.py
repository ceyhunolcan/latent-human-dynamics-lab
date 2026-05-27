"""Tests for the evaluation layer."""

import numpy as np


def test_regression_metrics_basic():
    from evaluation.metrics import regression_metrics

    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.1, 1.9, 3.0, 4.2])
    m = regression_metrics(y, yhat)
    assert "mae" in m and "rmse" in m and "r2" in m
    assert m["mae"] < 0.2


def test_classification_metrics_basic():
    from evaluation.metrics import classification_metrics

    y = np.array([0, 1, 2, 1, 0])
    yhat = np.array([0, 1, 2, 0, 0])
    m = classification_metrics(y, yhat)
    assert "accuracy" in m
    assert "f1_macro" in m


def test_calibration_curve_data_shape():
    from evaluation.calibration import calibration_curve_data, expected_calibration_error

    rng = np.random.RandomState(0)
    y = rng.binomial(1, 0.4, size=1000)
    p = rng.rand(1000)
    data = calibration_curve_data(y, p, n_bins=10)
    assert "prob_pred" in data and "prob_true" in data
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_missingness_stress_test_runs(engineered_cohort):
    from evaluation.robustness import missingness_stress_test

    df = engineered_cohort.copy()
    feature_cols = [c for c in ["sleep_duration_hours", "hrv_rmssd", "resting_hr"] if c in df.columns]
    # Fall back to actual present columns
    if not feature_cols:
        feature_cols = ["sleep_duration", "hrv_rmssd", "resting_hr"]
    df["__y__"] = (df["regime_label"] == "dysregulated").astype(int)

    def predict_fn(X):
        return np.zeros(len(X), dtype=int)

    result = missingness_stress_test(df, feature_cols, "__y__", predict_fn)
    assert len(result) >= 3


def test_synthetic_to_real_report_runs(small_cohort, engineered_cohort):
    from evaluation.synthetic_to_real import synthetic_to_real_report

    # Compare the cohort against itself — should yield near-identical scores
    a = small_cohort[["sleep_duration", "resting_hr"]].dropna()
    b = a.copy()
    report = synthetic_to_real_report(a, b)
    md = report.as_markdown()
    assert "Synthetic-to-real" in md or "similarity" in md.lower()


def test_logistic_baseline_trains_on_modern_sklearn():
    """Regression for Bug 10: LogisticRegression(multi_class='auto') was
    removed in sklearn 1.7+. Constructor should not pass that kwarg."""
    import numpy as np
    from models.downstream_tasks import train_logistic_baseline, evaluate_baseline

    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    head = train_logistic_baseline(X, y)
    m = evaluate_baseline(head, X, y)
    assert "accuracy" in m
    assert m["accuracy"] > 0.7
