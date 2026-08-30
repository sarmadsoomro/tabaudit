# tabaudit

A leakage-audit protocol for tabular regression: measures how much a
preprocessing step fit in the wrong order — on every row instead of on
training rows only — changes a model's reported performance.

Built for and released alongside the paper *"A Reusable Leakage-Audit
Protocol for Tabular Regression: Measuring Preprocessing-Leakage
Magnitude Across Three Housing Markets"* (Hakim & Soomro, submitted to
Data Mining and Knowledge Discovery).

## What this is (and isn't)

Three things, combined into one call:

1. **Matched arm construction** — build a "clean" (fit on training rows
   only) and a "leaky" (fit on every row) version of the same
   preprocessing step from one specification, so the only difference
   between them is which rows the fitted step saw.
2. **Paired cross-validation** — evaluate both arms on identical folds.
3. **Testing** — a plain paired significance test (is there a
   difference?) and a Nadeau-Bengio-corrected TOST equivalence test (are
   the two arms close enough to be practically interchangeable?).

**What is not claimed as novel:** the test layer alone. TOST equivalence
testing is already available in `statsmodels` and `pingouin`; the
Nadeau-Bengio correction for correlated CV folds is a well-known ten-line
formula. What this package adds is the *combination* — specifically the
arm-construction step, which those libraries do not provide.

## Install

```bash
pip install -e .          # from this directory
pip install -e ".[test]"  # + pytest, for the test suite
```

## Quickstart

```python
import pandas as pd
from tabaudit import run_audit

df = pd.read_csv("your_data.csv")

result = run_audit(
    df, target="price", mechanism="target_encoding_knn",
    enc_cols=["neighborhood"],
)

print(result["delta"])              # mean leaky-minus-clean R^2 difference
print(result["significance"]["p"])  # is there a difference at all?
print(result["is_equivalent"])      # are the two arms practically the same?
```

Three mechanisms are built in (`tabaudit.arms.MECHANISMS`):

| Mechanism | What it measures | Extra `run_audit` kwargs |
|---|---|---|
| `target_encoding_knn` | Target-encoding a categorical column + KNN imputation, fit on every row vs. training rows only | `enc_cols` |
| `group_aggregate` | A derived group-level aggregate feature (e.g. mean of some column per category), fit on every row vs. training rows only | `group_col`, `agg_col` |
| `feature_selection` | `SelectKBest(f_regression)` choosing columns using every row vs. training rows only | `k` |

For an already-computed pair of per-fold score arrays, skip arm
construction and call `audit_leakage(cv_clean, cv_leaky, n_train, n_test)`
directly.

## Non-housing demonstration

```bash
python3 examples/bike_sharing.py
```

Runs all three mechanisms against OpenML's `Bike_Sharing_Demand` (needs
network access on first run). Chosen because it has genuine
high-cardinality categoricals and a real grouping column, so it exercises
arm construction, not just the testing layer.

## Design notes

- Clean and leaky arms are built by the *same* function with different
  fit indices, never by two separate implementations — an asymmetric
  implementation would invalidate the whole comparison.
- No field returned by `audit_leakage`/`equivalence_test_tost` is
  pre-rounded. Round at your own display layer.
- Only Gradient Boosting and Random Forest are supported out of the box
  (`estimator_factory` accepts any `scikit-learn`-compatible regressor
  factory). Neural-network models are excluded because they are not
  bit-reproducible run-to-run.

## Development

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## License

MIT — see `LICENSE`.
