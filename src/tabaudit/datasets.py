"""Dataset loaders for demonstrations and tests.

``load_bike_sharing`` needs network access (fetches from OpenML) — it is
the non-housing worked demonstration (``examples/bike_sharing.py``), not
used by the test suite. ``make_synthetic_leakage_frame`` needs no network
and is used by the test suite instead.
"""

import numpy as np
import pandas as pd


def load_bike_sharing():
    """Load OpenML's Bike_Sharing_Demand dataset (version 2).

    Chosen for the non-housing demonstration because it has a genuine
    high-cardinality categorical (``hour``, 24 levels) and a plausible
    grouping column — unlike an all-numeric regression dataset, it
    exercises the ``target_encoding_knn`` and ``group_aggregate`` arms.

    Returns:
        (df, target, enc_cols, group_col, agg_col) — a dataframe and the
        column names ``run_audit`` needs for each of this package's three
        mechanisms.

    Raises:
        Whatever ``sklearn.datasets.fetch_openml`` raises on network
        failure.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True)
    df = bunch.frame.copy()
    target = "count"
    if target not in df.columns:
        # OpenML's frame sometimes exposes the target via bunch.target
        # instead of a same-named column — normalize either way.
        df[target] = bunch.target.values

    # `hour` (24 levels) is the highest-cardinality categorical available
    # here. `season`/`weather` (4 levels each) are too coarse for the
    # leaky/clean group mean to diverge meaningfully at this row count.
    enc_cols = ["hour"] if "hour" in df.columns else [c for c in ("season", "weather") if c in df.columns]
    group_col = "hour" if "hour" in df.columns else enc_cols[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_candidates = [c for c in ("temp", "humidity", "windspeed") if c in numeric_cols]
    agg_col = agg_candidates[0] if agg_candidates else numeric_cols[0]

    # OpenML's frame carries other categorical columns beyond enc_cols
    # (e.g. holiday/workingday, category-dtype booleans) that
    # target_encoding_knn's cast to float would choke on. Binary
    # categoricals become 0/1; anything else is dropped.
    for col in df.columns:
        if col in (target, *enc_cols) or col not in df.select_dtypes(exclude=[np.number]).columns:
            continue
        n_levels = df[col].nunique(dropna=False)
        if n_levels == 2:
            df[col] = df[col].astype("category").cat.codes.astype(float)
        else:
            df = df.drop(columns=[col])

    return df, target, enc_cols, group_col, agg_col


def make_synthetic_leakage_frame(n=500, n_groups=10, seed=0, leak_strength=0.0):
    """A small synthetic tabular-regression frame with a categorical
    column and a group structure, for offline tests and the ``_demo()``
    self-checks in this package's other modules.

    Args:
        n: Row count.
        n_groups: Distinct categories in the ``cat`` column.
        seed: RNG seed.
        leak_strength: If > 0, mixes a small amount of the target itself
            into a numeric feature — useful for a test that wants a
            *guaranteed* detectable leak rather than this arm's usual
            small, mechanism-specific effect.

    Returns:
        DataFrame with columns ``cat`` (categorical), ``num_a``, ``num_b``
        (numeric), ``age`` (numeric, for the group-aggregate arm),
        ``price`` (target).
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, size=n)
    price = 100 + 5 * group + rng.normal(scale=2, size=n)
    num_a = rng.normal(size=n) + leak_strength * price
    # `age` deliberately co-varies with `group` (hence with `price`) so the
    # group_aggregate mechanism has real information content to leak.
    age = 40 + 2 * group + rng.normal(scale=10, size=n)
    return pd.DataFrame({
        "cat": [f"g{g}" for g in group],
        "num_a": num_a,
        "num_b": rng.normal(size=n),
        "age": age,
        "price": price,
    })


def _demo():
    """Self-check: the synthetic frame has the columns every mechanism
    needs and no missing values."""
    df = make_synthetic_leakage_frame(n=200)
    assert set(df.columns) == {"cat", "num_a", "num_b", "age", "price"}
    assert df.isna().sum().sum() == 0
    assert df["cat"].nunique() <= 10
    print("demo: synthetic frame well-formed, no missing values. OK.")


if __name__ == "__main__":
    _demo()
