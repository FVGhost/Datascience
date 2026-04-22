# =============================================================================
# COM747 2025-2026 — Data Science & Machine Learning Coursework
# Dataset : COPD GOLD 230 Patients (Kaggle)
# Project : Classification of COPD Severity using GOLD Staging
# Task 1   : Data Loading, Cleaning & Preprocessing
=============================================================================
Member 1 : Naresh Balaji                  — B01047484
Member 2 : Osarobo Destiny Edomwande      — B01045758
Member 3 : Muhammed Mohsin                — B01028095
Member 4 : Aniket Yuvraj Chavan           — B01037393
Member 5 : Arun Reddy Bandi               — B01041589
=============================================================================

# CODE QUALITY PRINCIPLES:
#   • Constants    → updatability (one place to change)
#   • Comments     → readability for collaborators
#   • Functions    → reusability (DRY principle)
#   • RANDOM_STATE → reproducibility of all results
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH    = "230PatientsCOPD.xlsx"   # main dataset
CAT_PATH     = "PatientCategorical.csv" # categorical supplement
TARGET_COL   = "COPD GOLD"             # classification target
OUTPUT_DIR   = "outputs"               # all outputs saved here
RANDOM_STATE = 42                  # ensures reproducible results

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# TASK 1 — DATA LOADING, CLEANING & PREPROCESSING
# =============================================================================

print("=" * 65)
print("TASK 1 — DATA LOADING, CLEANING & PREPROCESSING")
print("=" * 65)

# ── 1.1 Load datasets ─────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
print(f"\nMain dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

if os.path.exists(CAT_PATH):
    df_cat = pd.read_csv(CAT_PATH)
    print(f"Categorical dataset loaded: {df_cat.shape[0]} rows × {df_cat.shape[1]} columns")
    # Merge on ID if present in both
    id_col_main = [c for c in df.columns if "ID" in c.upper()]
    id_col_cat  = [c for c in df_cat.columns if "ID" in c.upper()]
    if id_col_main and id_col_cat:
        df = pd.merge(df, df_cat,
                      left_on=id_col_main[0],
                      right_on=id_col_cat[0],
                      how="left", suffixes=("", "_cat"))
        print(f"Merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 1.2 Drop non-feature columns ──────────────────────────────────────────────
# Remove ID columns — they are identifiers, not predictive features.
# Keeping them would cause data leakage and inflate model performance.
id_cols = [c for c in df.columns if "ID" in c.upper() or "NUMBER" in c.upper()
           or c.strip().upper() == "ID"]
df = df.drop(columns=id_cols, errors="ignore")
print(f"\nDropped identifier columns: {id_cols}")
print(f"Working dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 1.3 Standardise column names ──────────────────────────────────────────────
# Strip whitespace and newline characters from column names
df.columns = df.columns.str.strip().str.replace("\n", "", regex=False)
print("\nCleaned column names:", df.columns.tolist())

# ── 1.4 Standardise string values ─────────────────────────────────────────────
# The dataset contains inconsistent capitalisation, trailing spaces,
# and special characters (e.g. 'Male ', 'male', 'NoN', '\xa0').
# These must be cleaned BEFORE encoding to avoid artificial extra categories.

def clean_string_column(series):
    """
    Standardises a string column by:
    - Stripping leading/trailing whitespace and non-breaking spaces
    - Converting to title case for consistency
    - Removing duplicate whitespace within values

    Parameters
    ----------
    series : pd.Series — column to clean

    Returns
    -------
    pd.Series — cleaned column
    """
    return (series
            .astype(str)
            .str.strip()
            .str.replace("\xa0", "", regex=False)  # remove non-breaking spaces
            .str.replace(r"\s+", " ", regex=True)  # collapse internal spaces
            .str.title()                             # consistent capitalisation
            .replace("Nan", np.nan))                # restore proper NaN

str_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in str_cols:
    df[col] = clean_string_column(df[col])
    print(f"  Cleaned string column: '{col}' → unique values: {df[col].dropna().unique().tolist()}")

# ── 1.5 Missing values ────────────────────────────────────────────────────────
print("\n── Missing Values ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]
print(missing_df if not missing_df.empty else "  No missing values found.")

# Drop rows where the target label is missing (can't train without a label)
before = len(df)
df = df.dropna(subset=[TARGET_COL])
print(f"\nDropped {before - len(df)} rows with missing target label.")

# Numerical columns: impute with median (robust to outliers)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if col == TARGET_COL:
        continue
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  '{col}': filled {n_missing} missing with median={median_val:.2f}")

# Categorical columns: impute with mode
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  '{col}': filled {n_missing} missing with mode='{mode_val}'")

# ── 1.6 Outlier detection and capping (IQR method) ───────────────────────────
# Winsorisation: cap outliers at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# This retains all rows while reducing the influence of extreme values.
print("\n── Outlier Detection & Capping (IQR Method) ──")
for col in num_cols:
    if col == TARGET_COL:
        continue
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  '{col}': {n_out} outliers capped to [{lower:.3f}, {upper:.3f}]")

# ── 1.7 Remove zero-variance columns ─────────────────────────────────────────
# Columns with only one unique value (e.g. Location = 'Barcelona' for all rows)
# carry no predictive information and can mislead models.
# Removing them reduces noise and improves model interpretability.
print("\n── Removing Zero-Variance Columns ──")
zero_var_cols = [col for col in df.columns
                 if col != TARGET_COL and df[col].nunique() <= 1]
if zero_var_cols:
    df = df.drop(columns=zero_var_cols)
    print(f"  Dropped {len(zero_var_cols)} zero-variance column(s): {zero_var_cols}")
else:
    print("  No zero-variance columns found.")

# Update column lists after dropping
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ── 1.8 Encode categorical variables ─────────────────────────────────────────
# Label encoding converts categories to integers for ML compatibility.
# Applied after cleaning so each category maps to exactly one integer.
from sklearn.preprocessing import LabelEncoder

print("\n── Encoding Categorical Variables ──")
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  '{col}' → {len(le.classes_)} classes: {le.classes_.tolist()}")

# ── 1.9 Final dataset summary ─────────────────────────────────────────────────
print(f"\nFinal clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Features retained: {[c for c in df.columns if c != TARGET_COL]}")
print("Class distribution (COPD GOLD):\n",
      df[TARGET_COL].value_counts().sort_index())

# ── 1.10 Save cleaned dataset ─────────────────────────────────────────────────
df.to_csv(f"{OUTPUT_DIR}/copd_cleaned.csv", index=False)
print(f"\nCleaned dataset saved → {OUTPUT_DIR}/copd_cleaned.csv")
print("\n✓ Task 1 complete.")
