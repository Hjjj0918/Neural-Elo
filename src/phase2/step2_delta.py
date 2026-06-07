"""
=============================================================================
 Phase 2, Step 2: CT-T Delta Features
=============================================================================
 Replace symmetric CT/T column pairs with delta = ct - t.
 Aggregate weapons by tier and compute tier deltas.
"""

import numpy as np
import pandas as pd


# ---- Symmetric CT-T pairs to merge into deltas (excluding weapons/grenades) ----
SYMMETRIC_PAIRS = [
    ("ct_score", "t_score"),
    ("ct_health", "t_health"),
    ("ct_armor", "t_armor"),
    ("ct_money", "t_money"),
    ("ct_helmets", "t_helmets"),
    ("ct_players_alive", "t_players_alive"),
]

# ---- Symmetric grenade pairs ----
GRENADE_PAIRS = [
    ("ct_grenade_hegrenade", "t_grenade_hegrenade"),
    ("ct_grenade_flashbang", "t_grenade_flashbang"),
    ("ct_grenade_smokegrenade", "t_grenade_smokegrenade"),
    ("ct_grenade_incendiarygrenade", "t_grenade_molotovgrenade"), # different names but same function
    ("ct_grenade_decoygrenade", "t_grenade_decoygrenade"),
]

# ---- Weapon tier definitions ----
# Maps tier name -> (CT columns, T columns)
WEAPON_TIERS = {
    "rifle": {
        "ct": ["ct_weapon_ak47", "ct_weapon_aug", "ct_weapon_famas",
               "ct_weapon_m4a1s", "ct_weapon_m4a4", "ct_weapon_sg553"],
        "t":  ["t_weapon_ak47", "t_weapon_aug", "t_weapon_galilar",
               "t_weapon_m4a4", "t_weapon_sg553"],
    },
    "awp": {
        "ct": ["ct_weapon_awp"],
        "t":  ["t_weapon_awp"],
    },
    "smg": {
        "ct": ["ct_weapon_mp9", "ct_weapon_ump45"],
        "t":  ["t_weapon_mac10", "t_weapon_ump45"],
    },
    "pistol": {
        "ct": ["ct_weapon_deagle", "ct_weapon_usps", "ct_weapon_p250",
               "ct_weapon_p2000", "ct_weapon_fiveseven", "ct_weapon_cz75auto"],
        "t":  ["t_weapon_deagle", "t_weapon_usps", "t_weapon_p250",
               "t_weapon_tec9", "t_weapon_glock", "t_weapon_cz75auto"],
    },
    "sniper_other": {
        "ct": ["ct_weapon_ssg08"],
        "t":  ["t_weapon_ssg08"],
    },
}


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform cleaned df: replace CT/T pairs with deltas,
    aggregate weapons into tier counts, produce tier deltas.

    Args:
        df: Cleaned DataFrame from Step 1 (contains CT/T columns)

    Returns:
        DataFrame with delta features only (original CT/T columns removed)
    """
    print("=" * 70)
    print("  Step 2: CT-T Delta Features")
    print("=" * 70)

    n_cols_before = len(df.columns)
    result = pd.DataFrame(index=df.index)

    # --- 2a. Core symmetric deltas ---
    deltas_created = []
    for ct_col, t_col in SYMMETRIC_PAIRS:
        if ct_col in df.columns and t_col in df.columns:
            base = ct_col[3:]  # strip "ct_" prefix
            result[f"delta_{base}"] = df[ct_col] - df[t_col]
            deltas_created.append(f"delta_{base}")
    print(f"\n  Core deltas ({len(deltas_created)}): {', '.join(deltas_created)}")

    # --- 2b. Grenade deltas ---
    g_deltas = []
    for ct_col, t_col in GRENADE_PAIRS:
        if ct_col in df.columns and t_col in df.columns:
            base = ct_col.replace("ct_grenade_", "")
            # Use both incendiary and molotov; sum them as CT side fires
            result[f"delta_{base}"] = df[ct_col] - df[t_col]
            g_deltas.append(f"delta_{base}")
    print(f"  Grenade deltas ({len(g_deltas)}): {', '.join(g_deltas)}")

    # --- 2c. Keep CT-only columns as-is ---
    ct_only = ["ct_defuse_kits"]
    for col in ct_only:
        if col in df.columns:
            result[col] = df[col]

    # --- 2d. Weapon tier aggregation + deltas ---
    w_deltas = []
    for tier_name, sides in WEAPON_TIERS.items():
        ct_cols = [c for c in sides["ct"] if c in df.columns]
        t_cols = [c for c in sides["t"] if c in df.columns]

        if ct_cols or t_cols:
            ct_sum = df[ct_cols].sum(axis=1) if ct_cols else 0
            t_sum = df[t_cols].sum(axis=1) if t_cols else 0
            result[f"delta_{tier_name}"] = ct_sum - t_sum
            w_deltas.append(f"delta_{tier_name}")

    print(f"  Weapon tier deltas ({len(w_deltas)}): {', '.join(w_deltas)}")

    # --- 2e. Aggregate grenade totals per side ---
    # Sum all CT grenades, all T grenades, then delta
    ct_g_all = [c for c in df.columns if c.startswith("ct_grenade_")]
    t_g_all = [c for c in df.columns if c.startswith("t_grenade_")]
    if ct_g_all and t_g_all:
        ct_g_total = df[ct_g_all].sum(axis=1)
        t_g_total = df[t_g_all].sum(axis=1)
        result["delta_utility_total"] = ct_g_total - t_g_total
        print(f"  Utility delta: delta_utility_total")

    # --- 2f. Carry through global columns (not CT/T-specific) ---
    GLOBAL_COLS = ["time_left", "map", "bomb_planted"]
    for col in GLOBAL_COLS:
        if col in df.columns:
            result[col] = df[col]
    print(f"  Retained global cols: {[c for c in GLOBAL_COLS if c in df.columns]}")

    n_cols_after = len(result.columns)
    print(f"\n  Columns: {n_cols_before} -> {n_cols_after} "
          f"(reduced {n_cols_before - n_cols_after}, {(1 - n_cols_after/n_cols_before)*100:.0f}%)")
    print(f"  [OK] Step 2 complete.")

    return result


if __name__ == "__main__":
    from src.phase2.step1_clean import run as clean
    df_clean, _ = clean()
    df_delta = run(df_clean)
    print(f"\nFinal delta columns:")
    for c in df_delta.columns:
        print(f"  {c}")
