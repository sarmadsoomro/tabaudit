"""Paired k-fold CV evaluation of a matched clean/leaky arm pair.

``paired_cv_scores`` and ``paired_cv_scores_both_arms`` are generic
primitives: given feature matrices that are already fully built, CV them.

``paired_cv_scores_matched`` builds the matrices fold-wise from a raw
dataframe via ``tabaudit.arms.build_arms``, calling the per-fold builder
once per fold with that fold's own training rows as the clean arm's fit
set — see ``tabaudit.arms`` for the builder contract.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from . import arms as _arms

DEFAULT_ESTIMATORS = {
    "GB": lambda seed: GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
        random_state=seed),
    "RF": lambda seed: RandomForestRegressor(
        n_estimators=300, min_samples_split=5, min_samples_leaf=2,
        random_state=seed, n_jobs=-1),
}


def paired_cv_scores(X, y, n_splits=10, seed=42, estimator_factory=None):
    """Per-fold R^2 scores for one already-built (X, y) under k-fold CV.

    Args:
        X, y: Feature matrix and target array.
        n_splits: Number of CV folds (default 10, matching this paper).
        seed: KFold shuffling seed.
        estimator_factory: Callable ``seed -> unfitted sklearn regressor``.
            Defaults to Gradient Boosting (see ``DEFAULT_ESTIMATORS``).

    Returns:
        List of per-fold R^2 scores, one per fold, in fold order.
    """
    if estimator_factory is None:
        estimator_factory = DEFAULT_ESTIMATORS["GB"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tri, vai in kf.split(X):
        model = estimator_factory(seed)
        model.fit(X[tri], y[tri])
        scores.append(float(r2_score(y[vai], model.predict(X[vai]))))
    return scores


def paired_cv_scores_both_arms(X_clean, X_leaky, y, n_splits=10, seed=42,
                                estimator_factory=None):
    """Per-fold R^2 for two already-built matrices under the *same* fold
    assignment — the pairing a paired test requires. ``X_clean``/
    ``X_leaky``/``y`` must share row order and length.

    This is the low-level primitive for a caller who has already built
    both arms some other way. If the arms need building from a raw
    dataframe via one of ``tabaudit.arms``'s mechanisms, use
    ``paired_cv_scores_matched`` instead.

    Returns:
        (cv_clean, cv_leaky) — two same-length lists of per-fold R^2,
        fold-for-fold comparable.
    """
    if len(X_clean) != len(X_leaky) or len(X_clean) != len(y):
        raise ValueError("X_clean, X_leaky, and y must be the same length (paired rows)")
    if estimator_factory is None:
        estimator_factory = DEFAULT_ESTIMATORS["GB"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_clean, cv_leaky = [], []
    for tri, vai in kf.split(X_clean):
        m_clean = estimator_factory(seed)
        m_clean.fit(X_clean[tri], y[tri])
        cv_clean.append(float(r2_score(y[vai], m_clean.predict(X_clean[vai]))))

        m_leaky = estimator_factory(seed)
        m_leaky.fit(X_leaky[tri], y[tri])
        cv_leaky.append(float(r2_score(y[vai], m_leaky.predict(X_leaky[vai]))))
    return cv_clean, cv_leaky


def paired_cv_scores_matched(df, target, mechanism, n_splits=10, seed=42,
                              estimator_factory=None, **arm_kwargs):
    """Per-fold R^2 for the clean (fold-wise-refit) and leaky (full-data-
    fit) arms of one leakage mechanism, built directly from a raw
    dataframe.

    Args:
        df: Source dataframe, including ``target``.
        target: Name of the target column.
        mechanism: One of ``tabaudit.arms.MECHANISMS``, or a custom
            callable — see ``tabaudit.arms.build_arms``.
        n_splits, seed: CV folds and KFold shuffle seed.
        estimator_factory: Callable ``seed -> unfitted sklearn regressor``.
            Defaults to Gradient Boosting.
        **arm_kwargs: Forwarded to ``tabaudit.arms.build_arms``.

    Returns:
        (cv_clean, cv_leaky) — two same-length lists of per-fold R^2,
        fold-for-fold comparable.
    """
    build_fold = _arms.build_arms(df, target, mechanism, **arm_kwargs)
    y = df[target].values.astype(float)
    n = len(df)
    all_idx = np.arange(n)
    if estimator_factory is None:
        estimator_factory = DEFAULT_ESTIMATORS["GB"]

    # The leaky arm's fit set (every row) is identical every fold — build
    # it once rather than refitting an identical transform n_splits times.
    X_leaky_full = build_fold(all_idx, all_idx)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_clean, cv_leaky = [], []
    for tri, vai in kf.split(all_idx):
        # Clean arm: transform fit on this fold's training rows only.
        X_clean_full = build_fold(tri, all_idx)
        yf, yv = y[tri], y[vai]

        m_clean = estimator_factory(seed)
        m_clean.fit(X_clean_full[tri], yf)
        cv_clean.append(float(r2_score(yv, m_clean.predict(X_clean_full[vai]))))

        m_leaky = estimator_factory(seed)
        m_leaky.fit(X_leaky_full[tri], yf)
        cv_leaky.append(float(r2_score(yv, m_leaky.predict(X_leaky_full[vai]))))
    return cv_clean, cv_leaky


def _demo():
    """Self-check: a synthetic leaky feature (a copy of the target, only
    visible to the leaky arm) should produce a large, obvious CV score
    gap; two identical arms should produce a near-zero gap. Also checks
    paired_cv_scores_matched end-to-end on a real mechanism."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(scale=3.0, size=n)

    cv_a, cv_b = paired_cv_scores_both_arms(X, X, y, n_splits=5)
    assert abs(np.mean(cv_a) - np.mean(cv_b)) < 1e-9, "identical arms must score identically"

    X_leaky = np.hstack([X, y.reshape(-1, 1)])  # target itself as a "leaked" feature
    cv_clean, cv_leaky = paired_cv_scores_both_arms(X, X_leaky, y, n_splits=5)
    assert np.mean(cv_leaky) > np.mean(cv_clean) + 0.3, (
        "a target-leaking feature should produce an obvious score gap")

    group = rng.integers(0, 8, size=n)
    df = pd.DataFrame({
        "group": [f"g{g}" for g in group],
        "num_a": rng.normal(size=n),
        "price": 100 + 5 * group + rng.normal(scale=2, size=n),
    })
    cv_c, cv_k = paired_cv_scores_matched(
        df, "price", "target_encoding_knn", enc_cols=["group"], n_splits=5)
    assert len(cv_c) == 5 and len(cv_k) == 5

    print("demo: identical arms score identically; a leaked target produces an obvious gap; "
          "paired_cv_scores_matched runs end-to-end. OK.")


if __name__ == "__main__":
    _demo()
