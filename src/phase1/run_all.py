"""
=============================================================================
 Phase 1: Run all EDA steps in sequence
=============================================================================

 Usage:
     python -m src.phase1.run_all
     python phase1_eda.py              # thin wrapper at project root
=============================================================================
"""

from .config import clean_output_dir
from .step0_find_data import run as find_data
from .step1_load import run as load_data
from .step2_missing import run as analyze_missing
from .step3_label import run as analyze_label
from .step4_numeric import run as analyze_numeric
from .step5_categorical import run as analyze_categorical
from .step6_correlation import run as analyze_correlation
from .step7_pitfalls import run as check_pitfalls
from .step8_summary import run as print_summary


def main():
    clean_output_dir()

    print("\n" + "#" * 70)
    print("#   Neural-Elo Phase 1: Exploratory Data Analysis (EDA)")
    print("#   CS:GO Round Winner Prediction")
    print("#" * 70)

    # 0. Locate data
    filepath = find_data()

    # 1. Load data
    df = load_data(filepath)

    # 2. Missing values
    missing_df = analyze_missing(df)

    # 3. Label
    label_col = analyze_label(df)

    # 4. Numeric features
    numeric_features = analyze_numeric(df, label_col)

    # 5. Categorical features
    cat_features = analyze_categorical(df, label_col)

    # 6. Correlation
    analyze_correlation(df, label_col, numeric_features)

    # 7. Data pitfalls
    check_pitfalls(df, label_col, numeric_features)

    # 8. Summary
    print_summary(df, label_col, numeric_features, cat_features, missing_df)

    print("\n" + "#" * 70)
    print("#   Phase 1 Complete! Ready for Phase 2.")
    print("#" * 70)


if __name__ == "__main__":
    main()
