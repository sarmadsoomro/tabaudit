"""Regression tests against the accompanying paper's published numbers.

Two independent checks:

1. ``audit_leakage``'s significance/d_z output reproduces Table 8's
   published GB/RF p-values and effect sizes exactly.
2. ``tabaudit.stats``'s functions produce output matching the original
   statistics module on the same inputs.

Both are skipped when the paper's ``code/`` directory is not present,
so this suite runs from a standalone checkout.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tabaudit.audit import audit_leakage

PAPER_CODE_DIR = Path(__file__).resolve().parents[2] / "code"

# Expected published values for Table 8 (GB/RF p-values and effect sizes).
PUBLISHED_TABLE8 = {
    ("Melbourne", "GB"): {"p": 0.035, "dz": 0.7827},
    ("Melbourne", "RF"): {"p": 0.038, "dz": None},
    ("Ames", "GB"): {"p": 0.431, "dz": 0.2608},
    ("Ames", "RF"): {"p": 0.343, "dz": None},
    ("King County", "GB"): {"p": 0.059, "dz": 0.6823},
    ("King County", "RF"): {"p": 0.221, "dz": None},
}


def _load_original_statistics_module():
    """Import the paper's ``code/statistics.py`` under an alias (never as
    the bare name ``statistics``, which would shadow the stdlib module)."""
    path = PAPER_CODE_DIR / "statistics.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_paper_original_statistics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset,model", [
    ("Melbourne", "gb"), ("Melbourne", "rf"),
    ("Ames", "gb"), ("Ames", "rf"),
    ("King County", "gb"), ("King County", "rf"),
])
def test_audit_leakage_reproduces_table8_significance(paper_checkpoint, dataset, model):
    """Exact reproduction of Table 8's published p-value and Cohen's d_z
    for all six GB/RF cells."""
    folds = paper_checkpoint[dataset][model]
    result = audit_leakage(folds["clean"], folds["leaky"], n_train=1, n_test=1)
    # n_train/n_test are irrelevant to the significance/d_z limb (plain
    # ttest_rel, no NB correction) — only the equivalence limb needs them.
    expected = PUBLISHED_TABLE8[(dataset, model.upper())]
    assert round(result["significance"]["p"], 3) == expected["p"]
    if expected["dz"] is not None:
        assert abs(result["d_z"] - expected["dz"]) < 1e-3


def test_ported_stats_module_matches_original_on_paper_data(paper_checkpoint):
    """Behavioral-equivalence check: run the original code/statistics.py
    functions and tabaudit's ported versions on identical inputs."""
    original = _load_original_statistics_module()
    if original is None:
        pytest.skip("parent paper's code/statistics.py not found")

    from tabaudit.stats import equivalence_test_tost as ported_tost
    from tabaudit.stats import nadeau_bengio_ttest as ported_nb

    folds = paper_checkpoint["King County"]["gb"]
    clean, leaky = folds["clean"], folds["leaky"]

    orig_t, orig_p = original.nadeau_bengio_ttest(leaky, clean, n_train=19061, n_test=2118)
    port_t, port_p = ported_nb(leaky, clean, n_train=19061, n_test=2118)
    # Original rounds internally to 4dp; the port does not — compare at
    # the original's own precision.
    assert abs(orig_t - port_t) < 5e-5
    assert abs(orig_p - port_p) < 5e-5

    diffs = leaky - clean
    delta = float(np.mean(diffs))
    orig_result = original.equivalence_test_tost(diffs, delta, n_train=19061, n_test=2118)
    port_result = ported_tost(diffs, delta, n_train=19061, n_test=2118)
    for key in ("delta", "se", "p_lower", "p_upper", "p_tost"):
        assert abs(orig_result[key] - port_result[key]) < 5e-5, key
    assert orig_result["is_equivalent"] == port_result["is_equivalent"]


def test_audit_leakage_matches_original_leakage_audit_module():
    """Equivalence check against the paper's original audit module."""
    path = PAPER_CODE_DIR / "leakage_audit.py"
    if not path.exists():
        pytest.skip("parent paper's code/leakage_audit.py not found")
    spec = importlib.util.spec_from_file_location("_paper_original_leakage_audit", path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    # The original module imports `code.statistics` — evict any cached
    # stdlib `code` module so Python resolves against the paper's path.
    saved_stdlib_code = sys.modules.pop("code", None)
    try:
        spec.loader.exec_module(original)
    finally:
        if saved_stdlib_code is not None:
            sys.modules["code"] = saved_stdlib_code
        else:
            sys.modules.pop("code", None)

    rng = np.random.default_rng(0)
    clean = np.array([0.84, 0.85, 0.83, 0.86, 0.84, 0.85, 0.83, 0.84, 0.86, 0.85])
    leaky = clean + rng.normal(0, 0.003, size=10)

    orig = original.audit_leakage(clean, leaky, n_train=900, n_test=100)
    port = audit_leakage(clean, leaky, n_train=900, n_test=100)

    assert abs(orig["delta"] - port["delta"]) < 1e-9
    assert abs(orig["significance"]["p"] - port["significance"]["p"]) < 1e-9
    assert abs(orig["d_z"] - port["d_z"]) < 1e-9
    assert orig["is_equivalent"] == port["is_equivalent"]
