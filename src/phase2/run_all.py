"""
=============================================================================
 Phase 2: Feature Engineering — Master Orchestrator
=============================================================================
 Runs all 7 steps in sequence, validates each, and saves output.

 Usage:
     python -m src.phase2.run_all
     python phase2_eda.py              # thin wrapper (to be created)
=============================================================================
"""

import json
import pickle
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, PROJECT_ROOT
from .step1_clean import run as step1_clean
from .step2_delta import run as step2_delta
from .step3_encode import run as step3_encode
from .step4_derive import run as step4_derive

warnings.filterwarnings("ignore", category=FutureWarning)


# ---- Column selection ----
# Columns to DROP from df_encoded when merging with derived features
# (time_left replaced by normalized time_remaining_pct from Step 4)
DROP_FROM_ENCODED = ["time_left"]


def assemble_features(df_encoded: pd.DataFrame, df_derived: pd.DataFrame) -> pd.DataFrame:
    """Merge encoded + derived features into a single X matrix."""
    # Drop redundant columns from encoded
    df_enc = df_encoded.drop(columns=[c for c in DROP_FROM_ENCODED if c in df_encoded.columns])
    # Concatenate
    X = pd.concat([df_enc, df_derived], axis=1)
    # Verify all-numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  [WARN] Non-numeric columns detected: {non_numeric}")
        # Attempt float conversion
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    # Check for any remaining NaN
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"NaN values found in feature matrix: {nan_cols}")
    return X


def scale_features(X_train, X_val, X_test, feature_names):
    """
    Fit StandardScaler on train only, transform all splits.

    CS:GO note: money and health have very different scales ($0-80000 vs 0-500 HP).
    Z-score normalization puts everything on comparable footing for neural networks.
    """
    print("\n" + "=" * 70)
    print("  Step 5: Standardization (StandardScaler)")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Scaler fit on train set ({X_train.shape[0]:,} samples)")
    print(f"  Feature count: {X_train_scaled.shape[1]}")

    # Quick sanity: check scaled means are ~0, stds are ~1 on train
    train_means = X_train_scaled.mean(axis=0)
    train_stds = X_train_scaled.std(axis=0)
    mean_max_abs = np.abs(train_means).max()
    std_max_abs = np.abs(train_stds - 1.0).max()
    print(f"  Train mean  (max |μ|):   {mean_max_abs:.6f}  (should be ~0)")
    print(f"  Train std   (max |σ-1|): {std_max_abs:.6f}  (should be ~0)")
    assert mean_max_abs < 1e-5, f"Scaled train means are not zero! max |μ| = {mean_max_abs}"
    assert std_max_abs < 1e-5, f"Scaled train stds are not 1! max |σ-1| = {std_max_abs}"

    print(f"  [OK] Step 5 complete.")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def split_data(X, y, random_state=42):
    """
    Train/val/test split: 70/15/15 stratified by label.

    No match_id available → rounds from same match may cross splits.
    We accept ~1-3% optimistic bias in test metrics.
    """
    print("\n" + "=" * 70)
    print("  Step 6: Train / Validation / Test Split")
    print("=" * 70)

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state, shuffle=True,
    )
    # Second split: 15% val, 15% test (from the 30% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp,
        random_state=random_state, shuffle=True,
    )

    print(f"\n  Split sizes:")
    print(f"    Train: {len(X_train):>8,}  ({len(X_train)/len(X)*100:.1f}%)  "
          f"CT={y_train.mean()*100:.1f}%")
    print(f"    Val:   {len(X_val):>8,}  ({len(X_val)/len(X)*100:.1f}%)  "
          f"CT={y_val.mean()*100:.1f}%")
    print(f"    Test:  {len(X_test):>8,}  ({len(X_test)/len(X)*100:.1f}%)  "
          f"CT={y_test.mean()*100:.1f}%")

    # Verify no leakage
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx), "Train/val overlap!"
    assert train_idx.isdisjoint(test_idx), "Train/test overlap!"
    assert val_idx.isdisjoint(test_idx), "Val/test overlap!"

    print(f"  [OK] No index overlap between splits.")
    print(f"  [OK] Step 6 complete.")

    return (X_train, X_val, X_test,
            y_train.to_numpy().ravel(), y_val.to_numpy().ravel(), y_test.to_numpy().ravel())


def save_outputs(X_tr, X_v, X_te, y_tr, y_v, y_te, feature_names, scaler):
    """Save all processed data and metadata to data/processed/."""
    print("\n" + "=" * 70)
    print("  Step 7: Save Processed Data")
    print("=" * 70)

    saved = []

    # Feature matrices
    for name, arr in [("X_train", X_tr), ("X_val", X_v), ("X_test", X_te)]:
        path = PROCESSED_DIR / f"{name}.npy"
        np.save(path, arr)
        saved.append(str(path))

    # Labels
    for name, arr in [("y_train", y_tr), ("y_val", y_v), ("y_test", y_te)]:
        path = PROCESSED_DIR / f"{name}.npy"
        np.save(path, arr.astype(np.int32))
        saved.append(str(path))

    # Feature names
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    saved.append(str(PROCESSED_DIR / "feature_names.json"))

    # Scaler
    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    saved.append(str(PROCESSED_DIR / "scaler.pkl"))

    print(f"\n  Saved {len(saved)} files to {PROCESSED_DIR}/")
    for p in saved:
        print(f"    {Path(p).name}")

    print(f"  [OK] Step 7 complete.")


def main():
    print("\n" + "#" * 70)
    print("#   Neural-Elo Phase 2: Feature Engineering")
    print("#   CS:GO Round Winner Prediction")
    print("#" * 70)

    # ---- Step 1: Clean ----
    df_clean, y = step1_clean()

    # ---- Step 2: Deltas ----
    df_delta = step2_delta(df_clean)

    # ---- Step 3: Encode ----
    df_encoded = step3_encode(df_delta)

    # ---- Step 4: Derived ----
    df_derived = step4_derive(df_clean)

    # ---- Assemble X ----
    print("\n" + "=" * 70)
    print("  Assemble Feature Matrix")
    print("=" * 70)
    X = assemble_features(df_encoded, df_derived)
    feature_names = X.columns.tolist()
    print(f"\n  Final X: {X.shape[0]:,} rows x {X.shape[1]} cols")
    print(f"  Original: 97 cols → Final: {X.shape[1]} cols "
          f"({(1 - X.shape[1]/97)*100:.0f}% reduction)")
    print(f"\n  Feature names:")
    for i, name in enumerate(feature_names):
        print(f"    [{i:2d}] {name}")

    # ---- Step 6: Split ----
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ---- Step 5: Scale (after split — fit on train only) ----
    X_train_s, X_val_s, X_test_s, scaler = scale_features(
        X_train, X_val, X_test, feature_names
    )

    # ---- Step 7: Save ----
    save_outputs(X_train_s, X_val_s, X_test_s,
                 y_train, y_val, y_test,
                 feature_names, scaler)

    print("\n" + "#" * 70)
    print("#   Phase 2 Complete! Ready for Phase 3 (Model Training).")
    print("#" * 70)


if __name__ == "__main__":
    main()
