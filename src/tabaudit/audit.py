"""End-to-end and standalone leakage-audit entry points.

``audit_leakage`` is the statistical layer alone, for a caller who already
has two paired per-fold score arrays. ``run_audit`` wires arm construction
(``tabaudit.arms``), paired CV evaluation (``tabaudit.evaluate``), and this
statistical layer together into one call.
"""

import numpy as np
from scipy import stats as scipy_stats

from . import evaluate as _evaluate
from .stats import equivalence_test_tost, tost_sensitivity_sweep


def audit_leakage(cv_scores_clean, cv_scores_leaky, n_train, n_test,
                   equiv_bound=0.01, bounds_sweep=(0.005, 0.01, 0.015)):
    """Measure and test the leakage effect between two paired per-fold
    score arrays from a leakage-free and a leaky variant of the same
    pipeline, evaluated on identical CV folds.

    Args:
        cv_scores_clean: Per-fold scores (e.g. R^2) for the leakage-free arm.
        cv_scores_leaky: Per-fold scores for the leaky arm — same folds,
            same order, same length as ``cv_scores_clean``.
        n_train: Per-fold training-set size (Nadeau-Bengio correction).
        n_test:  Per-fold test-set size (Nadeau-Bengio correction).
        equiv_bound: Practical-equivalence bound for the primary TOST test.
        bounds_sweep: Additional bounds for a TOST sensitivity sweep.

    Returns:
        Dict with:
          - ``delta``: mean paired difference (leaky - clean), full precision.
          - ``significance``: dict with ``t``, ``p`` from an *uncorrected*
            two-sided paired t-test (``scipy.stats.ttest_rel``) — "is
            there a difference?"
          - ``equivalence``: dict from ``equivalence_test_tost`` — a TOST
            verdict using a Nadeau-Bengio-corrected SE — "are the two arms
            close enough to be practically interchangeable?" A different
            question with a different, more conservative SE than
            ``significance``; do not substitute one test's p-value for
            the other's.
          - ``sweep``: TOST verdicts at each bound in ``bounds_sweep``.
          - ``d_z``: Cohen's d_z for the paired difference
            (``t / sqrt(n)``, from the uncorrected significance test).
        Every numeric field is full float precision — round at the
        display layer, not here (see ``tabaudit.stats`` module docstring).
    """
    clean = np.asarray(cv_scores_clean, dtype=float)
    leaky = np.asarray(cv_scores_leaky, dtype=float)
    if clean.shape != leaky.shape:
        raise ValueError(
            "cv_scores_clean and cv_scores_leaky must be the same length "
            f"(same folds, paired) — got shapes {clean.shape} and {leaky.shape}")

    diffs = leaky - clean
    delta = float(np.mean(diffs))
    n = len(diffs)

    t_stat, p_val = scipy_stats.ttest_rel(leaky, clean)
    d_z = float(t_stat) / np.sqrt(n)

    equivalence = equivalence_test_tost(
        diffs, delta, equiv_bound=equiv_bound, n_train=n_train, n_test=n_test)
    sweep = tost_sensitivity_sweep(
        diffs, delta, bounds=bounds_sweep, n_train=n_train, n_test=n_test)

    return {
        "delta": delta,
        "significance": {"t": float(t_stat), "p": float(p_val)},
        "d_z": float(d_z),
        "equivalence": equivalence,
        "sweep": sweep,
        "is_equivalent": bool(equivalence["is_equivalent"]),
    }


def run_audit(df, target, mechanism, n_splits=10, seed=42,
              equiv_bound=0.01, estimator_factory=None, **arm_kwargs):
    """Run a complete leakage audit: build the matched arms, evaluate both
    under paired CV, and test the result — the single call this package
    exists to provide.

    Args:
        df: Source dataframe, including ``target``.
        target: Name of the target column.
        mechanism: One of ``tabaudit.arms.MECHANISMS``.
        n_splits: CV folds.
        seed: KFold shuffle seed — fold assignment (and each fold's
            transform fit) is reproducible from this one seed.
        equiv_bound: TOST equivalence bound.
        estimator_factory: Callable ``seed -> unfitted sklearn regressor``
            (defaults to Gradient Boosting — see ``tabaudit.evaluate``).
        **arm_kwargs: Forwarded to ``tabaudit.arms.build_arms`` — see that
            function for the per-mechanism required keys (``enc_cols``,
            or ``group_col``/``agg_col``, or ``k``).

    Returns:
        The ``audit_leakage`` result dict, plus ``mechanism`` and
        ``n_rows`` for provenance.
    """
    cv_clean, cv_leaky = _evaluate.paired_cv_scores_matched(
        df, target, mechanism, n_splits=n_splits, seed=seed,
        estimator_factory=estimator_factory, **arm_kwargs)

    n_rows = len(df)
    kf_sizes = _fold_sizes(n_rows, n_splits, seed)
    result = audit_leakage(
        cv_clean, cv_leaky, n_train=kf_sizes[0], n_test=kf_sizes[1],
        equiv_bound=equiv_bound)
    result["mechanism"] = mechanism
    result["n_rows"] = n_rows
    result["cv_clean"] = cv_clean
    result["cv_leaky"] = cv_leaky
    return result


def _fold_sizes(n_rows, n_splits, seed):
    """(n_train, n_test) for one representative fold of a KFold(n_splits)
    split over ``n_rows`` rows — used for the Nadeau-Bengio correction.
    Fold sizes depend only on ``n_rows``/``n_splits`` (shuffling changes
    which rows land in a fold, not how many)."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tri, vai = next(kf.split(np.arange(n_rows)))
    return len(tri), len(vai)


def _demo():
    """Self-check: confirms the wrapper's significance test and TOST test
    each reproduce their standalone counterparts exactly, and that
    ``run_audit`` end-to-end detects an obvious synthetic leak."""
    import pandas as pd

    rng = np.random.default_rng(0)
    clean = np.array([0.84, 0.85, 0.83, 0.86, 0.84, 0.85, 0.83, 0.84, 0.86, 0.85])
    leaky_small = clean + rng.normal(0, 0.001, size=10)
    leaky_large = clean + rng.normal(0.05, 0.01, size=10)

    result_small = audit_leakage(clean, leaky_small, n_train=1000, n_test=100)
    result_large = audit_leakage(clean, leaky_large, n_train=1000, n_test=100)

    ref_t, ref_p = scipy_stats.ttest_rel(leaky_small, clean)
    assert abs(result_small["significance"]["t"] - float(ref_t)) < 1e-9
    assert abs(result_small["significance"]["p"] - float(ref_p)) < 1e-9
    assert result_small["is_equivalent"], "small, near-zero-mean diffs should confirm TOST equivalence"
    assert not result_large["is_equivalent"], "large, clearly-nonzero diffs should NOT confirm TOST equivalence"

    n = 500
    group = rng.integers(0, 10, size=n)
    df = pd.DataFrame({
        "cat": [f"g{g}" for g in group],
        "num": rng.normal(size=n),
        "price": 50 + 10 * group + rng.normal(scale=1.0, size=n),
    })
    audit = run_audit(df, "price", "target_encoding_knn", enc_cols=["cat"], n_splits=5)
    assert audit["mechanism"] == "target_encoding_knn"
    assert audit["n_rows"] == n
    assert "delta" in audit and "equivalence" in audit

    print("demo: audit_leakage matches scipy.stats.ttest_rel exactly; TOST behaves as expected. OK.")
    print("demo: run_audit end-to-end on a synthetic frame produced a well-formed result. OK.")


if __name__ == "__main__":
    _demo()
