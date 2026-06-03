"""
=============================================================================
 Phase 1, Step 1: Load data & inspect schema
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


def run(filepath: Path) -> pd.DataFrame:
    """Read CSV, print basic info, and return DataFrame."""
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

    _inspect_schema(df)
    return df


def _inspect_schema(df: pd.DataFrame) -> None:
    """Print column names, dtypes, and sample rows."""
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

    print(f"\n  Sample (first 5 rows):")
    print(df.head().to_string(max_colwidth=12))
    print(f"\n  ... ({len(df):,} total rows)")
