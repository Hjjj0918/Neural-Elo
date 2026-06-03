"""
=============================================================================
 Phase 1, Step 7: Data pitfalls check (no charts)
=============================================================================
"""

import pandas as pd


def run(df: pd.DataFrame, label_col: str, numeric_features: list) -> None:
    """Check for common data traps: leakage, duplicates, constant cols, etc."""
    print("\n" + "=" * 70)
    print("  Step 7: Data Pitfalls Check")
    print("=" * 70)

    pitfalls = 0
    pitfalls += _check_label_leakage(df, label_col)
    pitfalls += _check_temporal_leakage(numeric_features)
    pitfalls += _check_duplicates(df)
    pitfalls += _check_constant_cols(df, numeric_features)
    _check_match_id(df)
    _check_ct_t_symmetry(df)
    _print_pitfalls_summary(pitfalls)


def _check_label_leakage(df: pd.DataFrame, label_col: str) -> int:
    """Check if any column names overlap with label semantics."""
    print("\n  [1/6] Label leakage check...")
    suspicious = []
    for col in df.columns:
        if col != label_col:
            low = col.lower()
            if any(kw in low for kw in ["winner", "win_", "_win", "label", "target", "result"]):
                suspicious.append(col)
    if suspicious:
        print(f"    [WARN] Suspicious column names: {suspicious}")
        return 1
    print("    [OK] No suspicious column names found.")
    return 0


def _check_temporal_leakage(numeric_features: list) -> int:
    """Check for columns that might contain future information."""
    print("\n  [2/6] Temporal information leakage check...")
    future_like = [c for c in numeric_features
                   if any(kw in c.lower() for kw in ["total_rounds", "final", "end_"])]
    if future_like:
        print(f"    [WARN] Potential future-info columns: {future_like}")
        return 1
    print("    [OK] No obvious future information leakage detected.")
    return 0


def _check_duplicates(df: pd.DataFrame) -> int:
    """Check for fully duplicate rows."""
    print("\n  [3/6] Duplicate row check...")
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"    [WARN] {dup} fully duplicate rows ({dup/len(df)*100:.2f}%)")
        return 1
    print("    [OK] No fully duplicate rows.")
    return 0


def _check_constant_cols(df: pd.DataFrame, numeric_features: list) -> int:
    """Check for constant and near-constant columns."""
    print("\n  [4/6] Constant / quasi-constant column check...")
    p = 0
    constant = [c for c in numeric_features if df[c].nunique() <= 1]
    near_const = [c for c in numeric_features
                  if 1 < df[c].nunique() <= 5 and df[c].value_counts().iloc[0] / len(df) > 0.99]
    if constant:
        print(f"    [WARN] Constant columns (no info): {constant}")
        p += 1
    if near_const:
        print(f"    [WARN] Near-constant columns (>99% same value): {near_const}")
        p += 1
    if not constant and not near_const:
        print("    [OK] No constant or near-constant columns.")
    return p


def _check_match_id(df: pd.DataFrame) -> None:
    """Check if the dataset has a match/game ID column for GroupKFold."""
    print("\n  [5/6] Match ID column check...")
    id_cols = [c for c in df.columns
               if c.lower() in ("match_id", "matchid", "game_id", "gameid", "id")
               or c.lower().endswith("_id")]
    if id_cols:
        print(f"    [INFO] Potential grouping ID columns found: {id_cols}")
        print(f"    [TIP] Use GroupKFold (grouped by this column) instead of random split!")
    else:
        print("    [INFO] No match ID column. Standard KFold will be used (verify independence).")


def _check_ct_t_symmetry(df: pd.DataFrame) -> None:
    """Check for symmetric CT/T feature pairs."""
    print("\n  [6/6] CT/T feature symmetry check...")
    ct_cols = [c for c in df.columns if c.startswith("ct_")]
    t_cols = [c for c in df.columns if c.startswith("t_")]
    if ct_cols and t_cols:
        print(f"    [OK] Found symmetric CT/T features: {len(ct_cols)} CT cols, {len(t_cols)} T cols")
        print(f"    [TIP] CT-T difference features can be engineered (Phase 2).")
    else:
        print("    [INFO] No obvious CT/T symmetric features detected.")


def _print_pitfalls_summary(count: int) -> None:
    """Print pitfalls section summary."""
    print("\n" + "-" * 50)
    if count == 0:
        print("  [OK] No data pitfalls detected! Data quality looks excellent.")
    else:
        print(f"  [WARN] {count} potential issue(s) found. Address before Phase 2.")
