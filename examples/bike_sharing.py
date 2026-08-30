#!/usr/bin/env python3
"""Worked demonstration outside housing: OpenML's Bike_Sharing_Demand.

Chosen because it has a real high-cardinality categorical (hour of day,
24 levels) and a genuine grouping column, so it exercises the
target_encoding_knn and group_aggregate arms.

Needs network access on first run (OpenML fetch, cached by scikit-learn
afterwards). Not part of the test suite — see ``tests/`` for the
network-free checks.

Usage: python3 examples/bike_sharing.py
"""

from tabaudit import run_audit
from tabaudit.datasets import load_bike_sharing


def main():
    df, target, enc_cols, group_col, agg_col = load_bike_sharing()
    print(f"Bike_Sharing_Demand: {len(df)} rows, target={target!r}, "
          f"enc_cols={enc_cols}, group_col={group_col!r}, agg_col={agg_col!r}\n")

    results = {}
    results["target_encoding_knn"] = run_audit(
        df, target, "target_encoding_knn", enc_cols=enc_cols)
    results["group_aggregate"] = run_audit(
        df, target, "group_aggregate", group_col=group_col, agg_col=agg_col)
    results["feature_selection"] = run_audit(
        df, target, "feature_selection", k=8)

    print(f"{'mechanism':<22}{'delta':>10}{'p (sig.)':>10}{'p_tost':>10}{'equivalent?':>13}")
    for mechanism, r in results.items():
        print(f"{mechanism:<22}{r['delta']:>10.4f}{r['significance']['p']:>10.4f}"
              f"{r['equivalence']['p_tost']:>10.4f}{str(r['is_equivalent']):>13}")

    return results


if __name__ == "__main__":
    main()
