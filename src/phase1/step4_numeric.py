"""
=============================================================================
 Phase 1, Step 4: Numeric feature analysis  (Chart 02)
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import savefig


def run(df: pd.DataFrame, label_col: str) -> list:
    """Analyze numeric feature distributions. Returns list of feature column names."""
    print("\n" + "=" * 70)
    print("  Step 4: Numeric Feature Analysis")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c != label_col and not c.lower().endswith("_id") and c.lower() != "id"
    ]

    print(f"\n  Numeric feature count: {len(feature_cols)}")
    print(f"  (Excluded {len(numeric_cols) - len(feature_cols)} label/ID columns)\n")

    _print_statistics(df, feature_cols)
    _plot_distributions(df, feature_cols, label_col)

    return feature_cols


def _print_statistics(df: pd.DataFrame, feature_cols: list) -> None:
    """Print summary stats and flag problematic columns."""
    stats = df[feature_cols].describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / stats["mean"].abs().replace(0, np.nan)
    stats["skew"] = df[feature_cols].skew()
    stats["kurtosis"] = df[feature_cols].kurtosis()

    print("  Numeric feature statistics (first 20):")
    print(stats.round(2).to_string(max_rows=20))

    zero_var = stats[stats["std"] == 0].index.tolist()
    if zero_var:
        print(f"\n  [WARN] Zero-variance columns (recommend dropping):")
        print(f"         {zero_var}")

    skewed = stats[stats["skew"].abs() > 3].index.tolist()
    if skewed:
        print(f"\n  [INFO] Highly skewed columns (|skew| > 3, first 10):")
        print(f"         {skewed[:10]}")
        if len(skewed) > 10:
            print(f"         ... ({len(skewed) - 10} more)")


def _plot_distributions(df: pd.DataFrame, feature_cols: list, label_col: str) -> None:
    """Histogram grid of top 12 features by label correlation."""
    df_temp = df.copy()
    if df_temp[label_col].dtype == "object":
        df_temp["__label_num__"] = pd.factorize(df_temp[label_col])[0]
    else:
        df_temp["__label_num__"] = df_temp[label_col]

    valid = [c for c in feature_cols if df[c].nunique() > 1]
    corrs = df_temp[valid].corrwith(df_temp["__label_num__"]).abs()
    top_features = corrs.sort_values(ascending=False).head(12).index.tolist()

    n_cols, n_rows = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        ax.hist(df[feat].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(df[feat].median(), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"median={df[feat].median():.1f}")
        ax.axvline(df[feat].mean(), color="orange", linestyle="-.", linewidth=1.5,
                   label=f"mean={df[feat].mean():.1f}")
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    for j in range(len(top_features), n_rows * n_cols):
        axes[j].set_visible(False)

    plt.suptitle("Top 12 Features Distribution (by label correlation)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    savefig("02_numeric_feature_distributions")
