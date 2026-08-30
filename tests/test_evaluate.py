import numpy as np
import pandas as pd

from tabaudit.evaluate import paired_cv_scores, paired_cv_scores_both_arms, paired_cv_scores_matched


def test_paired_cv_scores_length_matches_n_splits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + rng.normal(scale=0.5, size=120)
    scores = paired_cv_scores(X, y, n_splits=5)
    assert len(scores) == 5


def test_identical_arms_score_identically():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    y = X.sum(axis=1) + rng.normal(scale=1.0, size=150)
    cv_a, cv_b = paired_cv_scores_both_arms(X, X, y, n_splits=5)
    assert np.allclose(cv_a, cv_b)


def test_leaked_feature_produces_higher_scores():
    rng = np.random.default_rng(3)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=3.0, size=n)
    X_leaky = np.hstack([X, y.reshape(-1, 1)])
    cv_clean, cv_leaky = paired_cv_scores_both_arms(X, X_leaky, y, n_splits=5)
    assert np.mean(cv_leaky) > np.mean(cv_clean)


def test_mismatched_lengths_raise():
    import pytest

    X = np.zeros((10, 2))
    X_short = np.zeros((5, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        paired_cv_scores_both_arms(X, X_short, y)


def test_custom_estimator_factory_is_used():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + rng.normal(scale=0.5, size=100)
    calls = []

    def factory(seed):
        calls.append(seed)
        return RandomForestRegressor(n_estimators=20, random_state=seed)

    scores = paired_cv_scores(X, y, n_splits=4, estimator_factory=factory)
    assert len(scores) == 4
    assert len(calls) == 4


def test_paired_cv_scores_matched_length_matches_n_splits(synthetic_frame):
    cv_clean, cv_leaky = paired_cv_scores_matched(
        synthetic_frame, "price", "target_encoding_knn", enc_cols=["cat"], n_splits=5)
    assert len(cv_clean) == 5
    assert len(cv_leaky) == 5


def test_paired_cv_scores_matched_detects_a_leaked_target_encoding():
    """With many small groups and sufficient noise, the leaky (full-data-
    fit) arm should score at least as well as the fold-wise-refit clean
    arm."""
    rng = np.random.default_rng(1)
    n = 400
    group = rng.integers(0, 80, size=n)
    df = pd.DataFrame({
        "cat": [f"g{g}" for g in group],
        "num_a": rng.normal(size=n),
        "price": 100 + 20 * group + rng.normal(scale=10.0, size=n),
    })
    cv_clean, cv_leaky = paired_cv_scores_matched(
        df, "price", "target_encoding_knn", enc_cols=["cat"], n_splits=5)
    assert np.mean(cv_leaky) >= np.mean(cv_clean), (
        "the leaky (full-data-fit) arm should score at least as well as the clean arm")
