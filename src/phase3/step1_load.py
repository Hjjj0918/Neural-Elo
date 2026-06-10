"""
=============================================================================
 Phase 3, Step 1: Load Processed Data
=============================================================================
"""

import json

import numpy as np

from .config import PROCESSED_DIR


def run() -> dict:
    """Load all Phase 2 output files. Returns a dict with all arrays."""
    print("=" * 70)
    print("  Step 1: Load Processed Data")
    print("=" * 70)

    data = {}

    for split in ["train", "val", "test"]:
        X = np.load(PROCESSED_DIR / f"X_{split}.npy")
        y = np.load(PROCESSED_DIR / f"y_{split}.npy")
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        print(f"  X_{split}: {X.shape[0]:,} rows x {X.shape[1]} cols, "
              f"y_{split}: {y.sum():,} CT ({y.mean()*100:.1f}%)")

    with open(PROCESSED_DIR / "feature_names.json") as f:
        data["feature_names"] = json.load(f)

    print(f"  Features: {len(data['feature_names'])}")
    print(f"  [OK] Step 1 complete.")
    return data


if __name__ == "__main__":
    run()
