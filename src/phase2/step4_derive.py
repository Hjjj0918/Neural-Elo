"""
=============================================================================
 Phase 2, Step 4: Derived Features
=============================================================================
 Compute features from raw team-level values (per-player averages, ratios).
 These need the raw CT/T columns BEFORE delta transformation.
"""

import numpy as np
import pandas as pd


def run(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from cleaned raw data.

    Args:
        df_clean: Cleaned DataFrame from Step 1 (still has raw CT/T columns)

    Returns:
        DataFrame of derived features (index aligned with df_clean).
    """
    print("=" * 70)
    print("  Step 4: Derived Features")
    print("=" * 70)

    result = pd.DataFrame(index=df_clean.index)
    derived = []

    # --- 4a. Time remaining (normalized) ---
    if "time_left" in df_clean.columns:
        result["time_remaining_pct"] = df_clean["time_left"] / 175.0 # convert time to 0 - 1
        derived.append("time_remaining_pct")
        print(f"  time_remaining_pct: range [{result['time_remaining_pct'].min():.2f}, "
              f"{result['time_remaining_pct'].max():.2f}]")

    # --- 4b. Alive ratio (CT share of living players) ---
    # Avoid division by zero in edge cases (both teams wiped simultaneously)
    total_alive = df_clean["ct_players_alive"] + df_clean["t_players_alive"]
    result["alive_ratio"] = np.where(
        total_alive > 0,
        df_clean["ct_players_alive"] / total_alive,
        0.5  # neutral when both have 0 alive (extremely rare)
    )
    derived.append("alive_ratio")
    print(f"  alive_ratio: range [{result['alive_ratio'].min():.3f}, "
          f"{result['alive_ratio'].max():.3f}]")

    # --- 4c. Per-player health (CT and T separately, then delta) ---
    ct_players = df_clean["ct_players_alive"].clip(lower=1)
    t_players = df_clean["t_players_alive"].clip(lower=1)

    result["health_per_player_ct"] = df_clean["ct_health"] / ct_players
    result["health_per_player_t"] = df_clean["t_health"] / t_players
    result["delta_health_per_player"] = (
        result["health_per_player_ct"] - result["health_per_player_t"]
    )
    derived.extend(["health_per_player_ct", "health_per_player_t", "delta_health_per_player"])
    print(f"  health_per_player_ct: mean={result['health_per_player_ct'].mean():.0f} HP")
    print(f"  health_per_player_t:  mean={result['health_per_player_t'].mean():.0f} HP")

    # --- 4d. Per-player money ---
    result["money_per_player_ct"] = df_clean["ct_money"] / ct_players
    result["money_per_player_t"] = df_clean["t_money"] / t_players
    result["delta_money_per_player"] = (
        result["money_per_player_ct"] - result["money_per_player_t"]
    )
    derived.extend(["money_per_player_ct", "money_per_player_t", "delta_money_per_player"])
    print(f"  money_per_player_ct:  mean=${result['money_per_player_ct'].mean():.0f}")
    print(f"  money_per_player_t:   mean=${result['money_per_player_t'].mean():.0f}")

    # --- 4e. Equipment score per player ---
    # Armor = 100 per player with vest, helmet = +100 value add
    # This is a rough proxy: armor_val + 100*helmet_count (per team, then /players)
    result["equip_score_ct"] = (df_clean["ct_armor"] + 100.0 * df_clean["ct_helmets"]) / ct_players
    result["equip_score_t"] = (df_clean["t_armor"] + 100.0 * df_clean["t_helmets"]) / t_players
    result["delta_equip_score"] = result["equip_score_ct"] - result["equip_score_t"]
    derived.extend(["equip_score_ct", "equip_score_t", "delta_equip_score"])
    print(f"  equip_score_ct: mean={result['equip_score_ct'].mean():.0f}")
    print(f"  equip_score_t:  mean={result['equip_score_t'].mean():.0f}")

    # --- 4f. Verify no NaN or Inf ---
    for col in derived:
        assert not result[col].isnull().any(), f"NaN in {col}!"
        assert not np.isinf(result[col]).any(), f"Inf in {col}!"

    print(f"\n  Derived features: {len(derived)}")
    print(f"  [OK] Step 4 complete.")

    return result


if __name__ == "__main__":
    from src.phase2.step1_clean import run as clean
    df_clean, _ = clean()
    df_derived = run(df_clean)
    print(f"\nFinal derived columns:")
    for c in df_derived.columns:
        print(f"  {c}: mean={df_derived[c].mean():.3f}, std={df_derived[c].std():.3f}")
