"""
=============================================================================
 Phase 1, Step 3: Label analysis  (Chart 01)
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import savefig


def run(df: pd.DataFrame) -> str:
    """Analyze label distribution. Auto-detects label column. Returns column name."""
    print("\n" + "=" * 70)
    print("  Step 3: Label Analysis")
    print("=" * 70)

    label_col = _detect_label(df)

    print(f"\n  Label column identified: '{label_col}'")
    raw_values = df[label_col].unique()
    print(f"  Unique values:  {sorted(raw_values, key=str)}")
    print(f"  Data type:      {df[label_col].dtype}")

    counts = df[label_col].value_counts()
    print(f"\n  Label distribution:")
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {str(val):>6s}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    minority_pct = counts.min() / counts.sum()
    print(f"\n  Minority class ratio: {minority_pct:.1%}")
    if 0.45 <= minority_pct <= 0.55:
        print("  [OK] Balanced labels. No special handling needed.")
    elif 0.35 <= minority_pct < 0.45:
        print("  [WARN] Mild imbalance. Consider class_weight='balanced'.")
    else:
        print("  [WARN] Severe imbalance. Consider SMOTE or weighted loss.")

    _plot_label(counts, label_col, df)
    return label_col


def _detect_label(df: pd.DataFrame) -> str:
    """Auto-detect the label column."""
    candidates = ["round_winner", "winner", "win", "label", "target", "round_win"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in ["winner", "win", "label", "target"]):
            return col

    print("\n  [ERROR] Could not auto-detect label column!")
    print(f"  Available columns: {sorted(df.columns.tolist())}")
    label_col = input("  Please enter label column name manually: ").strip()
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' does not exist!")
    return label_col


def _plot_label(counts: pd.Series, label_col: str, df: pd.DataFrame) -> None:
    """Bar + pie chart of label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    unique_vals = sorted(counts.index, key=str)
    palette = sns.color_palette("Set2", len(unique_vals))
    colors_map = dict(zip(unique_vals, palette))
    bar_colors = [colors_map[v] for v in counts.index]

    axes[0].bar(
        [str(v) for v in counts.index], counts.values,
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )
    axes[0].set_title(f"'{label_col}' Distribution (Count)", fontsize=13)
    axes[0].set_ylabel("Samples")
    # Place label inside the upper portion of each bar (white on dark, no overflow)
    for i, (val_name, val_count) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(
            i, val_count * 0.70,
            f"{val_name}\n{val_count:,}\n({val_count/len(df)*100:.1f}%)",
            ha="center", fontsize=10, fontweight="bold",
            va="center", color="white",
        )

    axes[1].pie(
        counts.values, labels=[str(v) for v in counts.index],
        autopct="%1.1f%%", colors=bar_colors, explode=[0.02] * len(counts),
    )
    axes[1].set_title(f"'{label_col}' Proportion", fontsize=13)

    plt.suptitle("Label Balance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig("01_label_distribution")
