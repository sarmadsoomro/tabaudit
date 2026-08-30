"""Paired significance and equivalence tests for CV-fold score arrays.

Values are returned at full precision — round at the display layer, not
here.
"""

import numpy as np
from scipy import stats


def nadeau_bengio_ttest(scores1, scores2, n_train, n_test):
    """Nadeau-Bengio corrected paired t-test.

    Accounts for the correlation between folds that share training rows
    when comparing cross-validation results.

    Args:
        scores1: Per-fold scores for model/arm 1.
        scores2: Per-fold scores for model/arm 2 (same folds, same order).
        n_train: Per-fold training-set size.
        n_test:  Per-fold test-set size.

    Returns:
        (t_statistic, p_value), full precision.
    """
    diff = np.asarray(scores1, dtype=float) - np.asarray(scores2, dtype=float)
    k = len(diff)
    mean_d = np.mean(diff)
    var_d = np.var(diff, ddof=1)
    correction = (1 / k) + (n_test / n_train)
    corrected = correction * var_d
    if corrected <= 0:
        return 0.0, 1.0
    t_stat = mean_d / np.sqrt(corrected)
    p_val = 2 * stats.t.sf(np.abs(t_stat), k - 1)
    return float(t_stat), float(p_val)


def equivalence_test_tost(diffs, delta, equiv_bound=0.01, n_train=None, n_test=None):
    """TOST equivalence test on paired fold-wise differences.

    Two One-Sided Tests (TOST) for practical equivalence.
    H0_lower: delta <= -equiv_bound
    H0_upper: delta >= +equiv_bound
    If both are rejected (p_tost < 0.05), the two arms are equivalent to
    within ``equiv_bound``.

    The standard error uses the Nadeau-Bengio correction for correlated CV
    folds (same formula as ``nadeau_bengio_ttest``) — a plain CV SE
    understates variance here, since folds share training rows.

    Args:
        diffs:       Per-fold differences (arm 2 minus arm 1, same fold order).
        delta:       Observed mean difference — pass ``np.mean(diffs)``.
        equiv_bound: Equivalence bound.
        n_train:     Per-fold training-set size (required, NB correction).
        n_test:      Per-fold test-set size (required, NB correction).

    Returns:
        Dict with delta, se, p_lower, p_upper, p_tost, is_equivalent —
        all full precision except ``is_equivalent`` (bool).
    """
    if n_train is None or n_test is None:
        raise ValueError("n_train and n_test are required for the Nadeau-Bengio correction")
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    var_d = np.var(diffs, ddof=1)
    correction = (1 / n) + (n_test / n_train)
    se = float(np.sqrt(correction * var_d))
    if se == 0.0:
        # Zero-variance diff (e.g. feature selection selects identical
        # columns in both arms). With zero variance, delta is known with
        # certainty, so the two TOST one-sided tests degenerate
        # deterministically instead of dividing by zero.
        equivalent = bool(-equiv_bound < delta < equiv_bound)
        p_tost = 0.0 if equivalent else 1.0
        return {
            "delta": float(delta),
            "se": 0.0,
            "p_lower": 0.0 if delta > -equiv_bound else 1.0,
            "p_upper": 0.0 if delta < equiv_bound else 1.0,
            "p_tost": p_tost,
            "is_equivalent": equivalent,
        }
    t_lower = (delta - (-equiv_bound)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
    t_upper = (delta - equiv_bound) / se
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    p_tost = max(p_lower, p_upper)
    return {
        "delta": float(delta),
        "se": se,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "is_equivalent": bool(p_tost < 0.05),
    }


def tost_sensitivity_sweep(diffs, delta, bounds=(0.005, 0.01, 0.015), n_train=None, n_test=None):
    """Run the TOST equivalence test at several bounds, so a reader is not
    limited to a single pre-specified bound.

    Returns:
        Dict keyed by bound, each value an ``equivalence_test_tost`` result.
    """
    return {
        b: equivalence_test_tost(diffs, delta, equiv_bound=b, n_train=n_train, n_test=n_test)
        for b in bounds
    }


def _demo():
    """Self-check: confirms the test functions produce well-formed output
    on synthetic inputs."""
    rng = np.random.default_rng(0)
    clean = np.array([0.84, 0.85, 0.83, 0.86, 0.84, 0.85, 0.83, 0.84, 0.86, 0.85])
    leaky = clean + rng.normal(0, 0.002, size=10)

    t, p = nadeau_bengio_ttest(leaky, clean, n_train=900, n_test=100)
    assert isinstance(t, float) and isinstance(p, float)

    diffs = leaky - clean
    delta = float(np.mean(diffs))
    result = equivalence_test_tost(diffs, delta, n_train=900, n_test=100)
    assert set(result) == {"delta", "se", "p_lower", "p_upper", "p_tost", "is_equivalent"}
    # No field is pre-rounded — full float precision.
    assert len(str(result["se"]).split(".")[-1]) > 4 or result["se"] == 0.0

    sweep = tost_sensitivity_sweep(diffs, delta, n_train=900, n_test=100)
    assert set(sweep) == {0.005, 0.01, 0.015}

    print("demo: nadeau_bengio_ttest, equivalence_test_tost, tost_sensitivity_sweep OK.")
    print("demo: no internal rounding confirmed. OK.")


if __name__ == "__main__":
    _demo()
