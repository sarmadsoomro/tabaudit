import numpy as np
import pytest

from tabaudit import MECHANISMS, run_audit


@pytest.mark.parametrize("mechanism,kwargs", [
    ("target_encoding_knn", {"enc_cols": ["cat"]}),
    ("group_aggregate", {"group_col": "cat", "agg_col": "age"}),
    ("feature_selection", {"k": 2}),
])
def test_run_audit_end_to_end(synthetic_frame, mechanism, kwargs):
    result = run_audit(synthetic_frame, "price", mechanism, n_splits=5, **kwargs)
    assert result["mechanism"] == mechanism
    assert result["n_rows"] == len(synthetic_frame)
    for key in ("delta", "significance", "d_z", "equivalence", "sweep", "is_equivalent"):
        assert key in result
    assert len(result["cv_clean"]) == 5
    assert len(result["cv_leaky"]) == 5


def test_run_audit_detects_a_synthetic_group_aggregate_leak():
    from tabaudit.datasets import make_synthetic_leakage_frame

    # Very small groups give the leaky arm's full-data group mean room to
    # diverge from the clean arm's per-fold, train-only mean.
    df = make_synthetic_leakage_frame(n=400, n_groups=150)
    result = run_audit(df, "price", "group_aggregate", group_col="cat", agg_col="age",
                        n_splits=5)
    assert result["delta"] > 0, "the leaky arm should score at least as well as the clean arm"
    assert result["significance"]["p"] < 0.05, "the gap should be statistically detectable"


def test_all_mechanisms_reachable_via_run_audit(synthetic_frame):
    kwargs_by_mechanism = {
        "target_encoding_knn": {"enc_cols": ["cat"]},
        "group_aggregate": {"group_col": "cat", "agg_col": "age"},
        "feature_selection": {"k": 2},
    }
    for mechanism in MECHANISMS:
        result = run_audit(synthetic_frame, "price", mechanism, n_splits=3,
                            **kwargs_by_mechanism[mechanism])
        assert np.isfinite(result["delta"])
