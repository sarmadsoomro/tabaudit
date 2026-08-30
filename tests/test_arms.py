import numpy as np
import pytest

from tabaudit.arms import MECHANISMS, build_arms


def test_all_mechanisms_registered():
    assert set(MECHANISMS) == {
        "target_encoding_knn", "group_aggregate", "feature_selection",
    }


@pytest.mark.parametrize("mechanism,kwargs", [
    ("target_encoding_knn", {"enc_cols": ["cat"]}),
    ("group_aggregate", {"group_col": "cat", "agg_col": "age"}),
    ("feature_selection", {"k": 2}),
])
def test_clean_and_leaky_folds_are_structurally_parallel(synthetic_frame, mechanism, kwargs):
    """The invariant the whole audit rests on: for the same fold split,
    clean and leaky builds must have identical shape and row order — they
    may only differ in which rows the fitted step saw."""
    build_fold = build_arms(synthetic_frame, "price", mechanism, **kwargs)
    n = len(synthetic_frame)
    all_idx = np.arange(n)
    fit_idx, apply_idx = all_idx[:240], all_idx[240:]

    X_clean = build_fold(fit_idx, apply_idx)
    X_leaky = build_fold(all_idx, apply_idx)
    assert X_clean.shape == X_leaky.shape
    assert X_clean.shape[0] == len(apply_idx)
    assert not np.isnan(X_clean).any()
    assert not np.isnan(X_leaky).any()


def test_target_encoding_knn_leaky_uses_full_data_mean(synthetic_frame):
    """Ground-truth check of the leak itself: fitting with fit_idx = every
    row must equal each row's group mean computed over *every* row, not
    just fit_idx's own subset — otherwise this isn't actually the
    mechanism the paper measures."""
    build_fold = build_arms(synthetic_frame, "price", "target_encoding_knn", enc_cols=["cat"])
    df = synthetic_frame.reset_index(drop=True)
    n = len(df)
    all_idx = np.arange(n)

    X_leaky = build_fold(all_idx, all_idx)
    full_group_means = df.groupby("cat")["price"].mean()
    raw_enc = df["cat"].map(full_group_means).values
    scaled_expected = (raw_enc - raw_enc.min()) / (raw_enc.max() - raw_enc.min())
    # The encoded column is appended last (after KNN-impute + scale, but
    # scaling is monotonic per-column so for a fully observed synthetic
    # frame, exact min-max-normalized value should match a hand-computed
    # min-max scale of the raw encoding).
    assert np.allclose(X_leaky[:, -1], scaled_expected, atol=1e-6)


def test_target_encoding_knn_clean_never_touches_holdout_rows(synthetic_frame):
    """A clean fold's encoder, fit on fit_idx only, must be independent
    of what's in the held-out rows — changing a held-out row's own target
    must not change that row's own encoded value."""
    df = synthetic_frame.reset_index(drop=True)
    n = len(df)
    all_idx = np.arange(n)
    fit_idx, holdout_idx = all_idx[:240], all_idx[240:]

    build_fold = build_arms(df, "price", "target_encoding_knn", enc_cols=["cat"])
    X_clean = build_fold(fit_idx, holdout_idx)
    X_leaky = build_fold(all_idx, holdout_idx)
    assert not np.allclose(X_clean[:, -1], X_leaky[:, -1])

    # Perturbing a held-out row's own target must not change its own
    # clean-arm encoded value — the clean encoder never saw it.
    df_perturbed = df.copy()
    df_perturbed.loc[holdout_idx, "price"] = df_perturbed.loc[holdout_idx, "price"] * 1000
    build_fold_perturbed = build_arms(df_perturbed, "price", "target_encoding_knn", enc_cols=["cat"])
    X_clean_perturbed = build_fold_perturbed(fit_idx, holdout_idx)
    assert np.allclose(X_clean[:, -1], X_clean_perturbed[:, -1]), (
        "a held-out row's own target leaked into its own clean-arm encoded value")


def test_unknown_mechanism_raises(synthetic_frame):
    with pytest.raises(ValueError):
        build_arms(synthetic_frame, "price", "not_a_real_mechanism")


def test_custom_callable_mechanism_is_accepted(synthetic_frame):
    """build_arms accepts a caller-supplied mechanism, not just the three
    named ones."""
    def custom_mechanism(df, target, **kwargs):
        def build_fold(fit_idx, apply_idx):
            X = df.drop(columns=[target]).select_dtypes(include=[np.number])
            return X.values.astype(float)[apply_idx]
        return build_fold

    build_fold = build_arms(synthetic_frame, "price", custom_mechanism)
    n = len(synthetic_frame)
    all_idx = np.arange(n)
    X = build_fold(all_idx, all_idx)
    assert X.shape[0] == n


def test_group_aggregate_shapes(synthetic_frame):
    n = len(synthetic_frame)
    all_idx = np.arange(n)
    fit_idx, apply_idx = all_idx[:240], all_idx[240:]
    build_fold = build_arms(synthetic_frame, "price", "group_aggregate",
                             group_col="cat", agg_col="age")
    Xc = build_fold(fit_idx, apply_idx)
    Xk = build_fold(all_idx, apply_idx)
    assert Xc.shape == Xk.shape


def test_feature_selection_k_effective_caps_at_available_columns(synthetic_frame):
    build_fold = build_arms(synthetic_frame, "price", "feature_selection", k=1000)
    n = len(synthetic_frame)
    all_idx = np.arange(n)
    n_numeric = synthetic_frame.drop(columns=["price"]).select_dtypes(include="number").shape[1]
    X = build_fold(all_idx, all_idx)
    assert X.shape[1] == n_numeric
