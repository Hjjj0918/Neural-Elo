"""
=============================================================================
 Phase 2, Step 1: Data Cleaning
=============================================================================
 Drop duplicates, zero-variance columns, and near-constant columns.
"""

import numpy as np
import pandas as pd

from .config import RAW_CSV

# Zero-variance weapon columns identified in Phase 1 EDA
# These are weapons the opposite faction cannot purchase:
# CT can't buy bizon, g3sg1, negev, r8revolver, sawedoff
# T can buy m249 but it's extremely rare and has no impact on model performance
ZERO_VAR_COLS = [
    "ct_weapon_bizon",
    "ct_weapon_g3sg1",
    "t_weapon_m249",
    "ct_weapon_negev",
    "ct_weapon_r8revolver",
    "ct_weapon_sawedoff",
]

# Near-constant threshold: >99% of values are 0
NEAR_CONST_THRESHOLD = 0.99

def identify_near_constant_cols(df: pd.DataFrame) -> list:
    """Find weapon columns where >99% of values are 0."""
    weapon_cols = [c for c in df.columns
                   if c.startswith("ct_weapon_") or c.startswith("t_weapon_")]
    near_const = []
    for col in weapon_cols:
        zero_pct = (df[col] == 0).mean()
        if zero_pct >= NEAR_CONST_THRESHOLD:
            near_const.append(col)
    return near_const


def run() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw data, perform cleaning, return (df_clean, y).

    Returns:
        df_clean: Cleaned feature DataFrame (labels not included)
        y: Label Series (1 = CT wins, 0 = T wins)
    """
    print("=" * 70)
    print("  Step 1: Data Cleaning")
    print("=" * 70)

    df = pd.read_csv(RAW_CSV)
    n_initial = len(df)
    n_cols_initial = len(df.columns)
    print(f"\n  Initial:  {n_initial:,} rows x {n_cols_initial} cols")

    # --- 1a. Drop duplicates ---
    df = df.drop_duplicates()
    n_dup = n_initial - len(df)
    print(f"  Dropped:  {n_dup:,} duplicate rows ({n_dup/n_initial*100:.2f}%)")

    # --- 1b. Extract labels ---
    y = (df["round_winner"].str.upper() == "CT").astype(int)
    df = df.drop(columns=["round_winner"])

    # --- 1c. Drop zero-variance columns ---
    zero_in_df = [c for c in ZERO_VAR_COLS if c in df.columns]
    if zero_in_df:
        df = df.drop(columns=zero_in_df)
        print(f"  Dropped:  {len(zero_in_df)} zero-variance weapon columns")

    # --- 1d. Drop near-constant columns ---
    near_const = identify_near_constant_cols(df)
    if near_const:
        df = df.drop(columns=near_const)
        print(f"  Dropped:  {len(near_const)} near-constant weapon columns (>99% zero)")

    # --- 1e. Verify no NaN ---
    assert df.isnull().sum().sum() == 0, "Unexpected NaN values after cleaning!"
    assert len(y) == len(df), "Label length mismatch!"

    n_final = len(df)
    n_cols_final = len(df.columns)
    print(f"\n  Final:    {n_final:,} rows x {n_cols_final} cols")
    print(f"  Labels:   CT={y.sum():,} ({y.mean()*100:.1f}%), "
          f"T={(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
    print(f"  [OK] Step 1 complete.")

    return df, y


if __name__ == "__main__":
    run()
