"""tabaudit — a leakage-audit protocol for tabular regression.

Measures the effect of preprocessing-fit-order leakage — fitting an
encoder, a derived aggregate, or a feature selector on data the model
will later be tested on, instead of on the training rows alone — by
building a matched leakage-free/leaky pair of pipelines from the same
specification and testing whether their cross-validated performance
differs.

Public API:
    run_audit(df, target, mechanism, **kwargs) -> dict
        End-to-end: build arms, evaluate under paired CV, test the result.
    audit_leakage(cv_scores_clean, cv_scores_leaky, n_train, n_test) -> dict
        The statistical layer alone, for already-computed CV scores.
    build_arms(df, target, mechanism, **kwargs) -> build_fold(fit_idx, apply_idx) -> X
        Arm construction alone — a per-fold builder, fit fresh on
        ``fit_idx`` each call.
    paired_cv_scores_matched(df, target, mechanism, **kwargs) -> (cv_clean, cv_leaky)
        Build both arms fold-wise from a raw dataframe and CV them.
    paired_cv_scores_both_arms(X_clean, X_leaky, y) -> (cv_clean, cv_leaky)
        Low-level: paired CV over two already-built matrices.

See the README for scope: this package does not claim novelty for the
*test* layer (TOST equivalence testing and its Nadeau-Bengio correction
are established methods, already available in statsmodels/pingouin) —
only for combining matched arm construction, paired CV, and that test
into one audit call.
"""

from .arms import MECHANISMS, build_arms
from .audit import audit_leakage, run_audit
from .evaluate import paired_cv_scores, paired_cv_scores_both_arms, paired_cv_scores_matched
from .stats import equivalence_test_tost, nadeau_bengio_ttest, tost_sensitivity_sweep

__version__ = "0.1.0"

__all__ = [
    "run_audit",
    "audit_leakage",
    "build_arms",
    "MECHANISMS",
    "paired_cv_scores",
    "paired_cv_scores_both_arms",
    "paired_cv_scores_matched",
    "equivalence_test_tost",
    "nadeau_bengio_ttest",
    "tost_sensitivity_sweep",
    "__version__",
]
