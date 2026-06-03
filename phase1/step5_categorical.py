"""
=============================================================================
 Phase 1, Step 5: Categorical feature analysis  (Chart 03)
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import savefig


def run(df: pd.DataFrame, label_col: str) -> list:
    """Analyze categorical variables (map, etc.). Returns list of cat column names."""
    print("\n" + "=" * 70)
    print("  Step 5: Categorical Feature Analysis")
    print("=" * 70)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != label_col]

    if not cat_cols:
        print("\n  [INFO] No categorical columns found.")
        return []

    print(f"\n  Categorical feature count: {len(cat_cols)}")
    print(f"  Columns: {cat_cols}\n")

    for col in cat_cols:
        _print_value_counts(df, col)

    if "map" in df.columns:
        _plot_map_analysis(df, label_col)

    return cat_cols


def _print_value_counts(df: pd.DataFrame, col: str) -> None:
    """Print value counts for a categorical column."""
    nunique = df[col].nunique()
    print(f"  {col}: {nunique} unique values")
    vc = df[col].value_counts()
    for val, cnt in vc.head(10).items():
        pct = cnt / len(df) * 100
        print(f"       {str(val):>20s}: {cnt:>7,}  ({pct:5.1f}%)")
    if nunique > 10:
        print(f"       ... ({nunique - 10} more values)")
    if nunique > 50:
        print(f"       [WARN] High-cardinality: One-Hot encoding would create {nunique} columns!")


def _plot_map_analysis(df: pd.DataFrame, label_col: str) -> None:
    """Side-by-side: map frequency + CT win rate per map."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: map frequency
    map_counts = df["map"].value_counts()
    sns.barplot(x=map_counts.values, y=map_counts.index, hue=map_counts.index,
                palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("Map Frequency (rounds)", fontsize=13)
    axes[0].set_xlabel("Rounds")

    # Right: CT win rate per map
    df_map = df.copy()
    if df_map[label_col].dtype == "object":
        df_map["_winner_is_ct"] = (df_map[label_col].str.upper() == "CT").astype(int)
    else:
        df_map["_winner_is_ct"] = df_map[label_col]

    map_winrate = (
        df_map.groupby("map")["_winner_is_ct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    map_winrate["mean"] *= 100
    map_winrate = map_winrate.sort_values("mean", ascending=True)

    colors = ["#FF6B6B" if v < 50 else "#4ECDC4" for v in map_winrate["mean"]]
    bars = axes[1].barh(
        map_winrate["map"], map_winrate["mean"],
        color=colors, edgecolor="black", linewidth=0.5,
    )
    axes[1].axvline(50, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].set_title("CT Win Rate by Map (%)", fontsize=13)
    axes[1].set_xlabel("CT Win Rate (%)")

    for bar, (_, row) in zip(bars, map_winrate.iterrows()):
        axes[1].text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.1f}%", va="center", fontsize=9,
        )

    plt.suptitle("Map Dimension Analysis", fontsize=14, fontweight="bold")
    # rect=[left,bottom,right,top]: reserve top 7% for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig("03_map_analysis")
