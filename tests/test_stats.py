import numpy as np
from scipy import stats as scipy_stats

from tabaudit.stats import equivalence_test_tost, nadeau_bengio_ttest, tost_sensitivity_sweep


def test_nadeau_bengio_reduces_to_hand_worked_example():
    # k=4 folds, tiny hand-checkable numbers.
    scores1 = np.array([0.80, 0.82, 0.78, 0.81])
    scores2 = np.array([0.79, 0.80, 0.77, 0.79])
    diff = scores1 - scores2
    k = len(diff)
    mean_d = diff.mean()
    var_d = diff.var(ddof=1)
    n_train, n_test = 900, 100
    correction = (1 / k) + (n_test / n_train)
    expected_t = mean_d / np.sqrt(correction * var_d)
    expected_p = 2 * scipy_stats.t.sf(abs(expected_t), k - 1)

    t, p = nadeau_bengio_ttest(scores1, scores2, n_train=n_train, n_test=n_test)
    assert abs(t - expected_t) < 1e-9
    assert abs(p - expected_p) < 1e-9


def test_nadeau_bengio_zero_variance_returns_safe_defaults():
    scores = np.array([0.8, 0.8, 0.8, 0.8])
    t, p = nadeau_bengio_ttest(scores, scores, n_train=900, n_test=100)
    assert t == 0.0
    assert p == 1.0


def test_equivalence_test_tost_matches_statsmodels_with_plain_se():
    """statsmodels' ttost_paired uses a plain (uncorrected) paired SE —
    confirm tabaudit's TOST math is the same two-one-sided-test procedure,
    by disabling the Nadeau-Bengio correction (n_test=0 makes the
    correction term (1/n)+(n_test/n_train) reduce to the plain 1/n
    variance-of-the-mean term statsmodels uses)."""
    import pytest

    statsmodels = pytest.importorskip("statsmodels")
    from statsmodels.stats.weightstats import ttost_paired

    rng = np.random.default_rng(1)
    x1 = rng.normal(0.85, 0.02, size=12)
    x2 = x1 + rng.normal(0, 0.005, size=12)
    bound = 0.03

    sm_pvalue, _, _ = ttost_paired(x1, x2, -bound, bound)

    diffs = x1 - x2
    delta = float(np.mean(diffs))
    n = len(diffs)
    result = equivalence_test_tost(diffs, delta, equiv_bound=bound, n_train=10**9, n_test=0)
    # n_test=0, n_train huge -> correction -> 1/n, matching a plain paired SE.
    assert abs(result["p_tost"] - sm_pvalue) < 1e-6


def test_equivalence_test_tost_requires_n_train_n_test():
    import pytest

    with pytest.raises(ValueError):
        equivalence_test_tost(np.array([0.01, -0.01]), 0.0)


def test_equivalence_test_tost_no_internal_rounding():
    """A caller must be able to trust these numbers are not already
    rounded, so its own rounding never double-rounds a boundary value."""
    diffs = np.array([0.01, 0.02, -0.005, 0.015, 0.0, 0.008, -0.002, 0.011, 0.006, 0.009])
    result = equivalence_test_tost(diffs, float(np.mean(diffs)), n_train=900, n_test=100)
    # A value that would be exactly 4dp by chance is vanishingly unlikely
    # across all five numeric fields simultaneously.
    numeric_fields = ("delta", "se", "p_lower", "p_upper", "p_tost")
    exactly_4dp = sum(
        1 for f in numeric_fields
        if round(result[f], 10) == round(result[f], 4)
    )
    assert exactly_4dp < len(numeric_fields)


def test_tost_sensitivity_sweep_keys_match_bounds():
    diffs = np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.005, -0.005, 0.015, -0.015, 0.008])
    sweep = tost_sensitivity_sweep(diffs, float(np.mean(diffs)),
                                    bounds=(0.005, 0.01, 0.015), n_train=900, n_test=100)
    assert set(sweep) == {0.005, 0.01, 0.015}
    for bound, res in sweep.items():
        assert res["is_equivalent"] in (True, False)


def test_wider_bound_never_less_likely_equivalent():
    """A looser equivalence bound can only make TOST's p_tost smaller or
    equal (easier to confirm), never larger — a monotonicity sanity check."""
    rng = np.random.default_rng(2)
    diffs = rng.normal(0.001, 0.01, size=10)
    delta = float(np.mean(diffs))
    p_tight = equivalence_test_tost(diffs, delta, equiv_bound=0.005, n_train=900, n_test=100)["p_tost"]
    p_loose = equivalence_test_tost(diffs, delta, equiv_bound=0.02, n_train=900, n_test=100)["p_tost"]
    assert p_loose <= p_tight + 1e-12
