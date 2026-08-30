import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "code" / "outputs" / "checkpoint_results.csv"
)


@pytest.fixture
def synthetic_frame():
    from tabaudit.datasets import make_synthetic_leakage_frame

    return make_synthetic_leakage_frame(n=400, n_groups=8, seed=0)


@pytest.fixture(scope="session")
def paper_checkpoint():
    """The accompanying paper's published fold arrays. Skipped if the
    paper's checkpoint isn't present, so this test suite runs in
    isolation from a standalone checkout."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"paper checkpoint not found at {CHECKPOINT_PATH}")
    df = pd.read_csv(CHECKPOINT_PATH)
    rows = {}
    for _, row in df.iterrows():
        name = row["Dataset"]
        rows[name] = {
            model: {
                "clean": np.array(ast.literal_eval(row[f"cv_{model}_folds"])),
                "leaky": np.array(ast.literal_eval(row[f"cv_{model}_lk_folds"])),
            }
            for model in ("gb", "rf")
        }
    return rows
