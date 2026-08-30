"""Matched leaky/clean preprocessing-arm construction.

Each builder returns a per-fold callable ``build_fold(fit_idx, apply_idx) -> X``,
fit fresh on whichever row indices the caller passes as ``fit_idx``. The
clean arm is fit on the fold's training rows only; the leaky arm is fit
on every row (the leak being measured).

Three mechanisms:

- ``target_encoding_knn`` — a high-cardinality categorical column is
  mean-target-encoded, remaining gaps KNN-imputed, all features min-max
  scaled. The leaky variant fits the encoding map, the imputer, and the
  scaler on every row; the clean variant fits all three on ``fit_idx``
  only.
- ``group_aggregate`` — a derived feature: the mean of some numeric column
  within each level of a grouping column (e.g. mean age within suburb).
  The leaky variant computes the group means from every row; the clean
  variant from ``fit_idx`` only.
- ``feature_selection`` — ``SelectKBest(f_regression)`` chooses which
  columns to keep. The leaky variant scores every row; the clean variant
  scores ``fit_idx`` only.

A caller may also pass a custom callable as ``mechanism`` instead of one of
the three names above — see ``build_arms``.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

MECHANISMS = ("target_encoding_knn", "group_aggregate", "feature_selection")


def _target_encoding_knn_builder(df, target, enc_cols, knn_k=5):
    """Factory for the ``target_encoding_knn`` mechanism. Returns
    ``build_fold(fit_idx, apply_idx) -> X``: mean-target-encode
    ``enc_cols`` (mapping computed from ``fit_idx`` rows only), then
    KNN-impute and MinMax-scale (also fit on ``fit_idx`` only), applied to
    ``apply_idx`` rows."""
    y = df[target].values.astype(float)
    X_raw_all = df.drop(columns=[target])

    def build_fold(fit_idx, apply_idx):
        global_mean = float(y[fit_idx].mean())
        X_raw = X_raw_all.copy()
        for col in enc_cols:
            if col not in X_raw.columns:
                continue
            mapping = df.iloc[fit_idx].groupby(col)[target].mean()
            X_raw[col + "_enc"] = X_raw[col].map(mapping).fillna(global_mean)
            X_raw = X_raw.drop(columns=[col])
        X_arr = X_raw.values.astype(float)
        knn = KNNImputer(n_neighbors=knn_k)
        scaler = MinMaxScaler()
        knn.fit(X_arr[fit_idx])
        scaler.fit(knn.transform(X_arr[fit_idx]))
        return scaler.transform(knn.transform(X_arr[apply_idx]))

    return build_fold


def _group_aggregate_builder(df, target, group_col, agg_col):
    """Factory for the ``group_aggregate`` mechanism. Returns
    ``build_fold(fit_idx, apply_idx) -> X``: a derived feature (mean of
    ``agg_col`` within each ``group_col`` level, computed from ``fit_idx``
    rows only) appended to the original numeric columns. No
    encoding/imputation/scaling — this arm isolates the aggregate-leakage
    mechanism alone, mirroring ``code/leakage_arms.py::
    run_derived_variable_arm``."""
    feat_name = f"{group_col}_avg_{agg_col}"
    X_raw_all = df.drop(columns=[target])

    def build_fold(fit_idx, apply_idx):
        mapping = df.iloc[fit_idx].groupby(group_col)[agg_col].mean()
        global_mean = float(df.iloc[fit_idx][agg_col].mean())
        X_raw = X_raw_all.copy()
        X_raw[feat_name] = df[group_col].map(mapping).fillna(global_mean)
        X_raw = X_raw.select_dtypes(include=[np.number])
        return X_raw.values.astype(float)[apply_idx]

    return build_fold


def _feature_selection_builder(df, target, k):
    """Factory for the ``feature_selection`` mechanism. Returns
    ``build_fold(fit_idx, apply_idx) -> X``: ``SelectKBest(f_regression)``
    fit on ``fit_idx`` rows' numeric columns only, applied (as a column
    mask) to ``apply_idx`` rows."""
    y = df[target].values.astype(float)
    X_raw = df.drop(columns=[target]).select_dtypes(include=[np.number])
    X_arr = X_raw.values.astype(float)
    k_eff = min(k, X_arr.shape[1])

    def build_fold(fit_idx, apply_idx):
        selector = SelectKBest(f_regression, k=k_eff)
        selector.fit(X_arr[fit_idx], y[fit_idx])
        support = selector.get_support()
        return X_arr[apply_idx][:, support]

    return build_fold


_BUILDER_FACTORIES = {
    "target_encoding_knn": _target_encoding_knn_builder,
    "group_aggregate": _group_aggregate_builder,
    "feature_selection": _feature_selection_builder,
}


def build_arms(df, target, mechanism, **kwargs):
    """Build a per-fold, matched-mechanism feature-matrix builder.

    Args:
        df: Source dataframe, including ``target``.
        target: Name of the target column.
        mechanism: One of ``tabaudit.arms.MECHANISMS``, OR a callable
            ``(df, target, **kwargs) -> build_fold`` implementing a custom
            leakage mechanism with the same per-fold-builder contract.
        **kwargs: Forwarded to the mechanism-specific factory — see
            ``_target_encoding_knn_builder`` (needs ``enc_cols``),
            ``_group_aggregate_builder`` (needs ``group_col``, ``agg_col``),
            ``_feature_selection_builder`` (needs ``k``).

    Returns:
        ``build_fold(fit_idx, apply_idx) -> X`` — a callable that fits the
        mechanism's transform on ``df.iloc[fit_idx]`` and returns the
        transformed feature matrix for ``df.iloc[apply_idx]``. A caller
        builds the "clean" arm of one CV fold with ``fit_idx`` = that
        fold's training row positions, and the "leaky" arm with
        ``fit_idx`` = every row position — see ``tabaudit.evaluate``.
    """
    if callable(mechanism):
        factory = mechanism
    elif mechanism in _BUILDER_FACTORIES:
        factory = _BUILDER_FACTORIES[mechanism]
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}, expected one of "
                          f"{MECHANISMS} or a callable")
    return factory(df, target, **kwargs)


def _demo():
    """Self-check: confirms the clean arm's encoder, fit only on a fold's
    training rows, differs from the leaky arm's encoder (fit on every row)
    on a synthetic frame with a known group structure, and that the clean
    arm never lets a row's own value leak into its own encoding."""
    rng = np.random.default_rng(0)
    n = 400
    group = rng.integers(0, 8, size=n)
    df = pd.DataFrame({
        "group": [f"g{g}" for g in group],
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
        "age": rng.normal(40, 10, size=n),
        "price": 100 + 5 * group + rng.normal(scale=2, size=n),
    })
    all_idx = np.arange(n)
    fit_idx = all_idx[:240]   # a stand-in "fold training set"
    apply_idx = all_idx[240:]  # a stand-in "fold test set"

    build_clean = build_arms(df, "price", "target_encoding_knn", enc_cols=["group"])
    Xc = build_clean(fit_idx, apply_idx)
    Xk = build_clean(all_idx, apply_idx)  # same builder, leaky fit_idx
    assert Xc.shape == Xk.shape
    assert not np.allclose(Xc, Xk), "leaky and clean target-encoding should differ"

    build_agg = build_arms(df, "price", "group_aggregate", group_col="group", agg_col="age")
    Xc2 = build_agg(fit_idx, apply_idx)
    Xk2 = build_agg(all_idx, apply_idx)
    assert Xc2.shape == Xk2.shape

    build_fs = build_arms(df, "price", "feature_selection", k=2)
    Xc3 = build_fs(fit_idx, apply_idx)
    assert Xc3.shape[1] == 2

    print("demo: all three mechanisms build fold-scoped, differing clean/leaky "
          "matrices from the same per-fold builder. OK.")


if __name__ == "__main__":
    _demo()
