"""
=============================================================================
 Phase 1, Step 6: Correlation analysis  (Charts 04, 05)
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import savefig


def run(df: pd.DataFrame, label_col: str, numeric_features: list) -> None:
    """Compute and visualize feature-label and inter-feature correlations."""
    print("\n" + "=" * 70)
    print("  Step 6: Correlation Analysis")
    print("=" * 70)

    df_corr = df[numeric_features].copy()

    if df[label_col].dtype == "object":
        df_corr["__target__"] = pd.factorize(df[label_col])[0]
    else:
        df_corr["__target__"] = df[label_col]

    valid_cols = [c for c in df_corr.columns if df_corr[c].nunique() > 1]
    df_corr = df_corr[valid_cols]
    print(f"\n  Features participating: {len(valid_cols)}")

    # Feature-label correlations
    target_corr = (
        df_corr.corr()["__target__"]
        .drop("__target__")
        .sort_values(key=abs, ascending=False)
    )

    _print_top_correlations(target_corr, label_col)
    _plot_feature_label_corr(target_corr, label_col)
    _plot_intercorrelation_heatmap(df_corr, target_corr)


def _print_top_correlations(target_corr: pd.Series, label_col: str) -> None:
    """Print top 20 features by correlation with label."""
    print(f"\n  Top 20 features by correlation with '{label_col}':")
    for feat, corr_val in target_corr.head(20).items():
        direction = "(+)" if corr_val > 0 else "(-)"
        bar = "#" * int(abs(corr_val) * 50)
        print(f"    {feat:>30s}: {corr_val:>+.4f}  {bar}  {direction}")


def _plot_feature_label_corr(target_corr: pd.Series, label_col: str) -> None:
    """Horizontal bar chart: top 20 feature-label correlations."""
    fig, ax = plt.subplots(figsize=(10, 7))
    top_20 = target_corr.head(20)
    colors = ["#4ECDC4" if v > 0 else "#FF6B6B" for v in top_20.values]
    sns.barplot(
        x=top_20.values, y=top_20.index, hue=top_20.index,
        palette=colors, legend=False, ax=ax, edgecolor="black", linewidth=0.3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f'Top 20 Features vs "{label_col}" Correlation', fontsize=14, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    savefig("04_feature_label_correlation")


def _plot_intercorrelation_heatmap(df_corr: pd.DataFrame, target_corr: pd.Series) -> None:
    """Heatmap: top 30 feature inter-correlation matrix."""
    top_30_feats = target_corr.head(30).index.tolist()
    corr_matrix = df_corr[top_30_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Top 30 Feature Inter-Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7, rotation=0)
    plt.tight_layout()
    savefig("05_feature_intercorrelation_heatmap")

    # Print highly correlated pairs
    print(f"\n  Highly correlated feature pairs (|r| > 0.95 -- potential redundancy):")
    found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.95:
                found = True
                print(f"    {corr_matrix.columns[i]:>30s} <-> {corr_matrix.columns[j]:<30s} r = {r:+.4f}")
    if not found:
        print("    [OK] No highly redundant feature pairs detected.")
