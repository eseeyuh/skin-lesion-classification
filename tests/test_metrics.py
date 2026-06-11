import numpy as np

from src.metrics import (compute_metrics, expected_calibration_error,
                         mcnemar_test, significance_stars)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    prob = np.eye(7)[y]
    m = compute_metrics(y, y, prob, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["auc_macro"] == 1.0
    assert set(m["per_class"]) == set(CLASSES)


def test_metrics_stay_in_unit_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, size=200)
    prob = rng.dirichlet(np.ones(7), size=200)
    m = compute_metrics(y, prob.argmax(1), prob, CLASSES)
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_macro"):
        assert 0.0 <= m[key] <= 1.0


def test_ece_bounds():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 7, size=500)
    prob = rng.dirichlet(np.ones(7), size=500)
    assert 0.0 <= expected_calibration_error(y, prob) <= 1.0


def test_ece_zero_when_perfectly_confident_and_correct():
    y = np.array([0, 1, 2, 3])
    prob = np.eye(7)[y]  # 100% confidence on the right class
    assert expected_calibration_error(y, prob) == 0.0


def test_mcnemar_counts_disagreements():
    labels = np.array([0, 0, 1, 1, 1])
    a = np.array([0, 0, 1, 1, 1])   # all correct
    b = np.array([1, 1, 0, 0, 0])   # all wrong
    r = mcnemar_test(a, b, labels)
    assert r["b1_a_only"] == 5
    assert r["b2_b_only"] == 0
    assert 0.0 <= r["pvalue"] <= 1.0


def test_significance_stars_thresholds():
    assert significance_stars(0.0005) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.2) == "ns"
