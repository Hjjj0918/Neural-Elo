"""
=============================================================================
 Phase 2, Step 3: Categorical Variable Encoding
=============================================================================
 One-Hot encode `map` (8 values → 8 binary columns).
 Keep `bomb_planted` as int (0/1).
"""

import numpy as np
import pandas as pd


# Expected map names (from Phase 1 EDA)
EXPECTED_MAPS = [
    "de_dust2", "de_inferno", "de_nuke", "de_mirage",
    "de_overpass", "de_train", "de_vertigo", "de_cache",
]


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns in the feature DataFrame.

    Args:
        df: DataFrame from Step 2 (contains delta features + map, bomb_planted, time_left)

    Returns:
        DataFrame with map One-Hot encoded, bomb_planted as int, map column removed.
    """
    print("=" * 70)
    print("  Step 3: Categorical Encoding")
    print("=" * 70)

    n_cols_before = len(df.columns)
    result = df.copy()

    # --- 3a. One-Hot encode map ---
    if "map" in result.columns:
        # Create map_ prefix columns
        map_dummies = pd.get_dummies(result["map"], prefix="map", dtype=int)

        # Ensure all expected maps exist (some rare maps may be absent in this slice)
        for m in EXPECTED_MAPS:
            col_name = f"map_{m}"
            if col_name not in map_dummies.columns:
                map_dummies[col_name] = 0

        result = pd.concat([result.drop(columns=["map"]), map_dummies], axis=1)
        print(f"\n  Map One-Hot: {list(map_dummies.columns)}")
    else:
        print(f"\n  [WARN] 'map' column not found — skipping One-Hot encoding")

    # --- 3b. bomb_planted as int ---
    if "bomb_planted" in result.columns:
        # True -> 1, False -> 0
        result["bomb_planted"] = result["bomb_planted"].astype(int)
        print(f"  bomb_planted: kept as 0/1 (mean={result['bomb_planted'].mean():.3f})") 
    else:
        print(f"  [WARN] 'bomb_planted' column not found")

    n_cols_after = len(result.columns)
    print(f"\n  Columns: {n_cols_before} -> {n_cols_after}")
    print(f"  [OK] Step 3 complete.")

    return result


if __name__ == "__main__":
    from src.phase2.step1_clean import run as clean
    from src.phase2.step2_delta import run as delta
    df_clean, _ = clean()
    df_delta = delta(df_clean)
    df_encoded = run(df_delta)
    print(f"\nFinal encoded columns:")
    for c in df_encoded.columns:
        print(f"  {c}")
