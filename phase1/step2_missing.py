"""
=============================================================================
 Phase 1, Step 2: Missing value analysis
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import savefig


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing value distribution. Returns missing_df."""
    print("\n" + "=" * 70)
    print("  Step 2: Missing Value Analysis")
    print("=" * 70)

    total = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total) * 100

    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
        "dtype": df.dtypes.values,
    })
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    if missing_df.empty:
        print("\n  [OK] No missing values found! Dataset is very clean.")
        return missing_df

    print(f"\n  [WARN] Found {len(missing_df)} columns with missing values:")
    print(missing_df.to_string(index=False))

    # Visualize
    cols_with_missing = missing_df["column"].tolist()
    if len(cols_with_missing) <= 30:
        fig, ax = plt.subplots(figsize=(12, max(4, len(cols_with_missing) * 0.3)))
        sns.heatmap(
            df[cols_with_missing].isnull().sample(min(2000, len(df)), axis=0),
            cbar=False, cmap="viridis", yticklabels=False, ax=ax,
        )
        ax.set_title("Missing Value Heatmap (2000-row sample)", fontsize=14)
        plt.tight_layout()
        savefig("missing_values_heatmap")

    return missing_df
