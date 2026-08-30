# Changelog

## 0.1.0 — 2026-08-29

Initial release.

- `tabaudit.arms`: matched clean/leaky arm construction for three
  mechanisms — `target_encoding_knn`, `group_aggregate`, `feature_selection`.
- `tabaudit.evaluate`: paired k-fold CV evaluation, GB/RF only.
- `tabaudit.stats`: Nadeau-Bengio corrected paired t-test and TOST
  equivalence test. No internal rounding; zero-variance-diff edge case
  guarded.
- `tabaudit.audit`: `audit_leakage` for pre-computed CV scores;
  `run_audit` for end-to-end audit from raw dataframe.
- `tabaudit.datasets`: `load_bike_sharing` (OpenML demonstration) and
  `make_synthetic_leakage_frame` (offline test fixture).
- Test suite: 37 tests covering arm construction, stats verification
  against `statsmodels`, and end-to-end smoke tests.
- `examples/bike_sharing.py`: non-housing worked demonstration.
- Packaging: MIT license, `pyproject.toml`, `CITATION.cff`.
