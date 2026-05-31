"""
=============================================================================
 Phase 1: CS:GO Round Winner Prediction -- Exploratory Data Analysis (EDA)
=============================================================================
 Dataset: CS:GO Round Winner Classification
 Source:  Kaggle (christianlillelund/csgo-round-winner-classification)
 Granularity: Round-level snapshots
 Target:    Predict round winner (CT side vs T side)

 Usage:
    1. Download dataset first:  python data/download_dataset.py
    2. Run this script:         python phase1_eda.py

 Output:
    - Terminal: data cleaning report, statistics, correlation analysis
    - Charts: label distribution, feature distributions, correlation plots
=============================================================================
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Global Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib global style -- try Chinese font, fallback to DejaVu Sans
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
sns.set_palette("Set2")

# Chart saving helper
_plot_counter = 0


def _savefig(name: str) -> str:
    """Save chart to outputs/phase1/ and return path."""
    global _plot_counter
    _plot_counter += 1
    path = OUTPUT_DIR / f"{_plot_counter:02d}_{name}.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"   [chart] Saved: {path}")
    return str(path)


# ============================================================
# Step 0: Locate data file
# ============================================================
def find_data_file() -> Path:
    """Find CSV in data/ directory."""
    if not DATA_DIR.exists():
        print("[ERROR] data/ directory not found. Please download dataset first:")
        print("        python data/download_dataset.py")
        sys.exit(1)

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files in data/. Please download dataset first:")
        print("        python data/download_dataset.py")
        sys.exit(1)

    preferred = DATA_DIR / "csgo_round_snapshots.csv"
    if preferred.exists():
        return preferred

    print(f"[WARN] csgo_round_snapshots.csv not found, using {csv_files[0].name}")
    return csv_files[0]


# ============================================================
# Step 1: Load data & basic info
# ============================================================
def step1_load_data(filepath: Path) -> pd.DataFrame:
    """Read CSV and print basic info."""
    print("=" * 70)
    print("  Step 1: Load Data")
    print("=" * 70)
    print(f"  File: {filepath}")

    df = pd.read_csv(filepath)
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    print(f"\n  [OK] Data loaded successfully!")
    print(f"       File size:     {file_size_mb:.1f} MB")
    print(f"       Shape:         {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"       Memory usage:  {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return df


def step1_inspect_schema(df: pd.DataFrame) -> None:
    """Print column names, dtypes, sample rows."""
    print("\n" + "-" * 50)
    print("  Schema & Data Types")
    print("-" * 50)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    other_cols = set(df.columns) - set(numeric_cols) - set(cat_cols)

    print(f"\n  Numeric columns ({len(numeric_cols)}):")
    for i in range(0, min(len(numeric_cols), 50), 5):
        chunk = list(numeric_cols)[i:i + 5]
        print(f"      {', '.join(chunk)}")
    if len(numeric_cols) > 50:
        print(f"      ... ({len(numeric_cols) - 50} more)")

    if len(cat_cols) > 0:
        print(f"\n  Categorical columns ({len(cat_cols)}):")
        for i in range(0, len(cat_cols), 5):
            chunk = list(cat_cols)[i:i + 5]
            print(f"      {', '.join(chunk)}")

    if other_cols:
        print(f"\n  Other type columns ({len(other_cols)}):")
        for c in other_cols:
            print(f"      {c}: {df[c].dtype}")

    # Sample rows
    print(f"\n  Sample (first 5 rows):")
    print(df.head().to_string(max_colwidth=12))
    print(f"\n  ... ({len(df):,} total rows)")


# ============================================================
# Step 2: Missing value analysis
# ============================================================
def step2_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing value distribution."""
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

    # Visualize missing values
    cols_with_missing = missing_df["column"].tolist()
    if len(cols_with_missing) <= 30:
        fig, ax = plt.subplots(figsize=(12, max(4, len(cols_with_missing) * 0.3)))
        sns.heatmap(
            df[cols_with_missing].isnull().sample(min(2000, len(df)), axis=0),
            cbar=False, cmap="viridis", yticklabels=False, ax=ax,
        )
        ax.set_title("Missing Value Heatmap (2000-row sample)", fontsize=14)
        plt.tight_layout()
        _savefig("missing_values_heatmap")
        plt.close()

    return missing_df


# ============================================================
# Step 3: Label analysis
# ============================================================
def step3_label_analysis(df: pd.DataFrame) -> str:
    """Analyze label column distribution, auto-detect label name."""
    print("\n" + "=" * 70)
    print("  Step 3: Label Analysis")
    print("=" * 70)

    # Auto-detect label column
    label_candidates = [
        "round_winner", "winner", "win", "label", "target", "round_win",
    ]

    label_col = None
    for candidate in label_candidates:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        for col in df.columns:
            low = col.lower()
            if any(kw in low for kw in ["winner", "win", "label", "target"]):
                label_col = col
                break

    if label_col is None:
        print("\n  [ERROR] Could not auto-detect label column!")
        print(f"  Available columns: {sorted(df.columns.tolist())}")
        label_col = input("  Please enter label column name manually: ").strip()
        if label_col not in df.columns:
            raise KeyError(f"Column '{label_col}' does not exist!")

    print(f"\n  Label column identified: '{label_col}'")

    # Label value analysis
    raw_values = df[label_col].unique()
    print(f"  Unique values:  {sorted(raw_values, key=str)}")
    print(f"  Data type:      {df[label_col].dtype}")

    counts = df[label_col].value_counts()
    print(f"\n  Label distribution:")
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {str(val):>6s}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    # Balance check
    minority_pct = counts.min() / counts.sum()
    print(f"\n  Minority class ratio: {minority_pct:.1%}")
    if 0.45 <= minority_pct <= 0.55:
        print("  [OK] Balanced labels. No special handling needed.")
    elif 0.35 <= minority_pct < 0.45:
        print("  [WARN] Mild imbalance. Consider class_weight='balanced'.")
    else:
        print("  [WARN] Severe imbalance. Consider SMOTE or weighted loss.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    unique_vals = sorted(counts.index, key=str)
    palette = sns.color_palette("Set2", len(unique_vals))
    colors_map = dict(zip(unique_vals, palette))
    bar_colors = [colors_map[v] for v in counts.index]

    axes[0].bar(
        [str(v) for v in counts.index],
        counts.values,
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )
    axes[0].set_title(f"'{label_col}' Distribution (Count)", fontsize=13)
    axes[0].set_ylabel("Samples")
    for i, (val_name, val_count) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(
            i, val_count + len(df) * 0.005,
            f"{val_name}\n{val_count:,}\n({val_count/len(df)*100:.1f}%)",
            ha="center", fontsize=10, fontweight="bold",
        )

    axes[1].pie(
        counts.values,
        labels=[str(v) for v in counts.index],
        autopct="%1.1f%%",
        colors=bar_colors,
        explode=[0.02] * len(counts),
    )
    axes[1].set_title(f"'{label_col}' Proportion", fontsize=13)

    plt.suptitle("Label Balance Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig("01_label_distribution")
    plt.close()

    return label_col


# ============================================================
# Step 4: Numeric feature analysis
# ============================================================
def step4_numeric_features(df: pd.DataFrame, label_col: str) -> list:
    """Analyze distribution of all numeric features."""
    print("\n" + "=" * 70)
    print("  Step 4: Numeric Feature Analysis")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude label and ID columns
    feature_cols = [
        c for c in numeric_cols
        if c != label_col
        and not c.lower().endswith("_id")
        and not c.lower() == "id"
    ]

    print(f"\n  Numeric feature count: {len(feature_cols)}")
    print(f"  (Excluded {len(numeric_cols) - len(feature_cols)} label/ID columns)\n")

    # Statistics
    stats = df[feature_cols].describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / stats["mean"].abs().replace(0, np.nan)
    stats["skew"] = df[feature_cols].skew()
    stats["kurtosis"] = df[feature_cols].kurtosis()

    print("  Numeric feature statistics (first 20):")
    print(stats.round(2).to_string(max_rows=20))

    # Detect problematic columns
    zero_variance_cols = stats[stats["std"] == 0].index.tolist()
    if zero_variance_cols:
        print(f"\n  [WARN] Zero-variance columns (recommend dropping):")
        print(f"         {zero_variance_cols}")

    extremely_skewed = stats[stats["skew"].abs() > 3].index.tolist()
    if extremely_skewed:
        print(f"\n  [INFO] Highly skewed columns (|skew| > 3, first 10):")
        print(f"         {extremely_skewed[:10]}")
        if len(extremely_skewed) > 10:
            print(f"         ... ({len(extremely_skewed) - 10} more)")

    # Select top features by label correlation for visualization
    df_temp = df.copy()
    if df_temp[label_col].dtype == "object":
        df_temp["__label_num__"] = pd.factorize(df_temp[label_col])[0]
    else:
        df_temp["__label_num__"] = df_temp[label_col]

    valid_for_corr = [c for c in feature_cols if df[c].nunique() > 1]
    corrs = df_temp[valid_for_corr].corrwith(df_temp["__label_num__"]).abs()
    top_features = corrs.sort_values(ascending=False).head(12).index.tolist()

    # Distribution histograms (Top 12 features)
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

    plt.suptitle("Top 12 Features Distribution (by label correlation)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("02_numeric_feature_distributions")
    plt.close()

    return feature_cols


# ============================================================
# Step 5: Categorical feature analysis
# ============================================================
def step5_categorical_features(df: pd.DataFrame, label_col: str) -> list:
    """Analyze categorical variables (maps, etc.)."""
    print("\n" + "=" * 70)
    print("  Step 5: Categorical Feature Analysis")
    print("=" * 70)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != label_col]

    if not cat_cols:
        print("\n  [INFO] No categorical columns found.")
        print("   (Some dataset versions use numeric encoding for all variables)")
        return []

    print(f"\n  Categorical feature count: {len(cat_cols)}")
    print(f"  Columns: {cat_cols}\n")

    for col in cat_cols:
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

    # Map distribution visualization
    if "map" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        map_counts = df["map"].value_counts()
        sns.barplot(x=map_counts.values, y=map_counts.index, palette="viridis", ax=axes[0])
        axes[0].set_title("Map Frequency (rounds)", fontsize=13)
        axes[0].set_xlabel("Rounds")

        # CT win rate per map
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
        plt.tight_layout()
        _savefig("03_map_analysis")
        plt.close()

    return cat_cols


# ============================================================
# Step 6: Correlation analysis
# ============================================================
def step6_correlation_analysis(df: pd.DataFrame, label_col: str, numeric_features: list):
    """Compute and visualize feature correlations."""
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

    # Label correlation
    target_corr = (
        df_corr.corr()["__target__"]
        .drop("__target__")
        .sort_values(key=abs, ascending=False)
    )

    print(f"\n  Top 20 features by correlation with '{label_col}':")
    for feat, corr_val in target_corr.head(20).items():
        direction = "(+)" if corr_val > 0 else "(-)"
        bar = "#" * int(abs(corr_val) * 50)
        print(f"    {feat:>30s}: {corr_val:>+.4f}  {bar}  {direction}")

    # Bar chart: Top 20
    fig, ax = plt.subplots(figsize=(10, 7))
    top_20 = target_corr.head(20)
    colors = ["#4ECDC4" if v > 0 else "#FF6B6B" for v in top_20.values]
    sns.barplot(
        x=top_20.values, y=top_20.index,
        palette=colors, ax=ax, edgecolor="black", linewidth=0.3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Top 20 Features vs '{label_col}' Correlation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    _savefig("04_feature_label_correlation")
    plt.close()

    # Feature inter-correlation heatmap (Top 30)
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
    _savefig("05_feature_intercorrelation_heatmap")
    plt.close()

    # Detect highly correlated feature pairs
    print(f"\n  Highly correlated feature pairs (|r| > 0.95 -- potential redundancy):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r))
                print(f"    {corr_matrix.columns[i]:>30s} <-> {corr_matrix.columns[j]:<30s} r = {r:+.4f}")

    if not high_corr_pairs:
        print("    [OK] No highly redundant feature pairs detected.")


# ============================================================
# Step 7: Data pitfalls check
# ============================================================
def step7_data_pitfalls(df: pd.DataFrame, label_col: str, numeric_features: list):
    """Check for common data traps."""
    print("\n" + "=" * 70)
    print("  Step 7: Data Pitfalls Check")
    print("=" * 70)

    pitfalls_found = 0

    # Check 1: Label leakage
    print("\n  [1/6] Label leakage check...")
    suspicious_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col != label_col:
            leak_keywords = ["winner", "win_", "_win", "label", "target", "result"]
            if any(kw in col_lower for kw in leak_keywords):
                suspicious_cols.append(col)
    if suspicious_cols:
        print(f"    [WARN] Suspicious column names: {suspicious_cols}")
        pitfalls_found += 1
    else:
        print("    [OK] No suspicious column names found.")

    # Check 2: Future information leakage
    print("\n  [2/6] Temporal information leakage check...")
    future_like_cols = []
    for col in numeric_features:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["total_rounds", "final", "end_"]):
            future_like_cols.append(col)
    if future_like_cols:
        print(f"    [WARN] Potential future-info columns: {future_like_cols}")
        pitfalls_found += 1
    else:
        print("    [OK] No obvious future information leakage detected.")

    # Check 3: Duplicate rows
    print("\n  [3/6] Duplicate row check...")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"    [WARN] {dup_count} fully duplicate rows ({dup_count/len(df)*100:.2f}%)")
        pitfalls_found += 1
    else:
        print("    [OK] No fully duplicate rows.")

    # Check 4: Constant/quasi-constant columns
    print("\n  [4/6] Constant / quasi-constant column check...")
    constant_cols = [c for c in numeric_features if df[c].nunique() <= 1]
    near_constant_cols = [
        c for c in numeric_features
        if 1 < df[c].nunique() <= 5 and df[c].value_counts().iloc[0] / len(df) > 0.99
    ]
    if constant_cols:
        print(f"    [WARN] Constant columns (no info): {constant_cols}")
        pitfalls_found += 1
    if near_constant_cols:
        print(f"    [WARN] Near-constant columns (>99% same value): {near_constant_cols}")
        pitfalls_found += 1
    if not constant_cols and not near_constant_cols:
        print("    [OK] No constant or near-constant columns.")

    # Check 5: Match ID column
    print("\n  [5/6] Match ID column check...")
    id_cols = [
        c for c in df.columns
        if c.lower() in ("match_id", "matchid", "game_id", "gameid", "id")
        or c.lower().endswith("_id")
    ]
    if id_cols:
        print(f"    [INFO] Potential grouping ID columns found: {id_cols}")
        print(f"    [TIP] Use GroupKFold (grouped by this column) instead of random split!")
    else:
        print("    [INFO] No match ID column. Standard KFold will be used (verify independence).")

    # Check 6: CT/T feature symmetry
    print("\n  [6/6] CT/T feature symmetry check...")
    ct_cols = [c for c in df.columns if c.startswith("ct_")]
    t_cols = [c for c in df.columns if c.startswith("t_")]
    if ct_cols and t_cols:
        print(f"    [OK] Found symmetric CT/T features: {len(ct_cols)} CT cols, {len(t_cols)} T cols")
        print(f"    [TIP] CT-T difference features can be engineered (Phase 2).")
    else:
        print("    [INFO] No obvious CT/T symmetric features detected.")

    # Summary
    print("\n" + "-" * 50)
    if pitfalls_found == 0:
        print("  [OK] No data pitfalls detected! Data quality looks excellent.")
    else:
        print(f"  [WARN] {pitfalls_found} potential issue(s) found. Address before Phase 2.")


# ============================================================
# Step 8: Summary report
# ============================================================
def step8_summary_report(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: list,
    cat_features: list,
    missing_df: pd.DataFrame,
):
    """Print final EDA summary."""
    print("\n" + "=" * 70)
    print("  Step 8: EDA Summary Report")
    print("=" * 70)

    n_total = len(df)
    n_features = len(numeric_features) + len(cat_features)

    # Build label distribution string
    label_dist = dict(df[label_col].value_counts().to_dict())

    summary = f"""
  +-------------------------------------------------------------------+
  |                    EDA SUMMARY REPORT                              |
  +-------------------------------------------------------------------+
  |  Dataset:        CS:GO Round Winner Classification                 |
  |  Samples:        {n_total:>10,}                                            |
  |  Total features: {n_features:>10,}                                            |
  |    - Numeric:    {len(numeric_features):>10,}                                            |
  |    - Categorical:{len(cat_features):>10,}                                            |
  |  Label column:   {label_col:<35s}   |
  |  Label dist:     {str(label_dist):<35s}   |
  |  Missing cols:   {len(missing_df):>10,}                                            |
  |  Duplicate rows: {df.duplicated().sum():>10,}                                            |
  |  Memory usage:   {df.memory_usage(deep=True).sum()/1024**2:.1f} MB                                    |
  +-------------------------------------------------------------------+
  |  Charts saved to: {str(OUTPUT_DIR):<35s}   |
  +-------------------------------------------------------------------+

  Phase 1 EDA complete!
  Next: Phase 2 -- Feature Engineering
    - Construct CT-T delta aggregation features
    - Standardization & One-Hot Encoding
    - Train/Test split (pay attention to grouping!)
  """
    print(summary)

    # Save as text file
    report_path = OUTPUT_DIR / "eda_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Report saved: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "#" * 70)
    print("#   Neural-Elo Phase 1: Exploratory Data Analysis (EDA)")
    print("#   CS:GO Round Winner Prediction")
    print("#" * 70)

    # 0. Locate data
    filepath = find_data_file()

    # 1. Load data
    df = step1_load_data(filepath)
    step1_inspect_schema(df)

    # 2. Missing values
    missing_df = step2_missing_values(df)

    # 3. Label
    label_col = step3_label_analysis(df)

    # 4. Numeric features
    numeric_features = step4_numeric_features(df, label_col)

    # 5. Categorical features
    cat_features = step5_categorical_features(df, label_col)

    # 6. Correlation
    step6_correlation_analysis(df, label_col, numeric_features)

    # 7. Data pitfalls
    step7_data_pitfalls(df, label_col, numeric_features)

    # 8. Summary
    step8_summary_report(df, label_col, numeric_features, cat_features, missing_df)

    print("\n" + "#" * 70)
    print("#   Phase 1 Complete! Ready for Phase 2.")
    print("#" * 70)


if __name__ == "__main__":
    main()
