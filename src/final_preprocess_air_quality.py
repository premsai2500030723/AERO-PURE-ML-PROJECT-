"""
final_preprocess_air_quality.py
================================
Applies the **complete end-to-end preprocessing pipeline** to the raw
Aotizhongxin air quality dataset and saves a single production-ready CSV.

Pipeline Steps
--------------
1. Load raw dataset
2. Remove duplicate rows
3. Fill numeric NaN → column median
4. Fill categorical NaN → column mode
5. Strip & lowercase categorical strings
6. Label encode categorical columns (wd, station)
7. Standard scale all numeric columns (mean=0, std=1)
8. Save final preprocessed CSV

Output
------
dataset/final_preprocess_air_quality.csv

Usage
-----
    python src/final_preprocess_air_quality.py

Notes
-----
- The original raw CSV is never overwritten.
- StandardScaler is fit on the full dataset here; in a production pipeline
  it should be fit only on the training split to avoid data leakage.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==========================================================
# FILE PATHS
# ==========================================================

input_file = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

output_file = r"C:\Users\pream\PycharmProjects\ML- project\dataset\final_preprocess_air_quality.csv"


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv(input_file)

processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)


# ==========================================================
# 2. REMOVE DUPLICATE RECORDS
# ==========================================================

processed_df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:", processed_df.shape)


# ==========================================================
# 3. FILL MISSING NUMERICAL VALUES WITH MEDIAN
# ==========================================================

numeric_cols = processed_df.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_cols:

    if processed_df[col].isnull().any():

        median_value = processed_df[col].median()

        processed_df[col] = processed_df[col].fillna(median_value)


# ==========================================================
# 4. FILL MISSING CATEGORICAL VALUES WITH MODE
# ==========================================================

categorical_cols = processed_df.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:

    if processed_df[col].isnull().any():

        mode_value = processed_df[col].mode()[0]

        processed_df[col] = processed_df[col].fillna(mode_value)


# ==========================================================
# 5. STRIP SPACES AND LOWERCASE CATEGORICAL COLUMNS
# ==========================================================

for col in categorical_cols:

    processed_df[col] = processed_df[col].astype(str).str.strip().str.lower()


# ==========================================================
# 6. LABEL ENCODING FOR CATEGORICAL COLUMNS
# ==========================================================

for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = encoder.fit_transform(processed_df[col])


# ==========================================================
# 7. STANDARD SCALING FOR NUMERICAL COLUMNS
# ==========================================================

scaler = StandardScaler()

processed_df[numeric_cols] = scaler.fit_transform(
    processed_df[numeric_cols]
)


# ==========================================================
# 8. SAVE FINAL PREPROCESSED DATASET
# ==========================================================

processed_df.to_csv(output_file, index=False)


# ==========================================================
# 9. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("Preprocessing Pipeline Completed Successfully ?")
print("=" * 60)

print("\nOriginal Shape:  ", df.shape)
print("Processed Shape: ", processed_df.shape)
print("\nSaved File Path:")
print(output_file)

