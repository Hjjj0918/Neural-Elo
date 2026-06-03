"""
=============================================================================
 Phase 1, Step 8: Summary report (no charts)
=============================================================================
"""

import pandas as pd

from .config import OUTPUT_DIR


def run(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: list,
    cat_features: list,
    missing_df: pd.DataFrame,
) -> None:
    """Print final EDA summary report and save to text file."""
    print("\n" + "=" * 70)
    print("  Step 8: EDA Summary Report")
    print("=" * 70)

    n_total = len(df)
    n_features = len(numeric_features) + len(cat_features)
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

    report_path = OUTPUT_DIR / "eda_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Report saved: {report_path}")
