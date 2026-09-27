"""
clean_minmax_stand_norma_air_quality.py
=========================================
Loads the cleaned air quality dataset and applies three scaling strategies:

  1. **StandardScaler**  — zero mean, unit variance  (z-score normalisation)
  2. **MinMaxScaler**    — rescales features to [0, 1]
  3. **Normalizer**      — scales each sample to unit norm (L2)

Output
------
dataset/clean_air_quality_scaling_M2.csv   (StandardScaler result)

Usage
-----
    python src/clean_minmax_stand_norma_air_quality.py

Notes
-----
- Input is the Method-2 cleaned CSV (no categorical columns).
- Comparison plots are saved to the output/ folder.
- StandardScaler result is the primary output used by ML models.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)

import matplotlib.pyplot as plt


# ==========================================================
# FILE PATHS
# ==========================================================

file_path = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

output_file = r"C:\Users\pream\PycharmProjects\ML- project\dataset\clean_minmax_stand_norm_air_quality.csv"


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv(file_path)

data = df.copy()

print("=" * 80)
print("AIR QUALITY DATASET")
print("=" * 80)

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())


# ==========================================================
# 2. REMOVE DUPLICATES
# ==========================================================

data = data.drop_duplicates()

print("\nRows After Removing Duplicates:")
print(len(data))


# ==========================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ==========================================================

numerical_columns = data.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = data.select_dtypes(
    exclude="number"
).columns.tolist()

print("\n" + "=" * 80)
print("COLUMN TYPES")
print("=" * 80)

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ==========================================================
# 4. FILL MISSING NUMERICAL VALUES WITH MEAN
# ==========================================================

print("\n" + "=" * 80)
print("MISSING VALUE HANDLING")
print("=" * 80)

for col in numerical_columns:

    if data[col].isnull().any():

        mean_value = data[col].mean()

        missing_count = data[col].isnull().sum()

        data[col] = data[col].fillna(mean_value)

        print(
            f"{col}: "
            f"{missing_count} missing values "
            f"filled with mean = {mean_value:.4f}"
        )


# ==========================================================
# 5. FILL MISSING CATEGORICAL VALUES WITH MODE
# ==========================================================

for col in categorical_columns:

    if data[col].isnull().any():

        mode_values = data[col].mode()

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

            missing_count = data[col].isnull().sum()

            data[col] = data[col].fillna(mode_value)

            print(
                f"{col}: "
                f"{missing_count} missing values "
                f"filled with mode = {mode_value}"
            )


# ==========================================================
# 6. REMOVE EXTRA SPACES FROM CATEGORICAL COLUMNS
# ==========================================================

print("\n" + "=" * 80)
print("TEXT CLEANING")
print("=" * 80)

for col in categorical_columns:
    data[col] = data[col].astype(str).str.strip()

print("Extra spaces removed from categorical columns.")


# ==========================================================
# 7. GET NUMERIC COLUMNS AFTER CLEANING
# ==========================================================

numeric_columns = data.select_dtypes(
    include="number"
).columns.tolist()

print("\n" + "=" * 80)
print("NUMERIC COLUMNS SELECTED FOR SCALING")
print("=" * 80)

print(numeric_columns)


# ==========================================================
# 8. STANDARDIZATION
# ==========================================================

print("\n" + "=" * 80)
print("1. STANDARDIZATION")
print("=" * 80)

standard_scaler = StandardScaler()

standardized_data = standard_scaler.fit_transform(
    data[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    data[col + "_Standardized"] = standardized_data[:, i]

print("Standardization completed.")
print("Formula: (X - Mean) / Standard Deviation")


# ==========================================================
# 9. MIN-MAX SCALING
# ==========================================================

print("\n" + "=" * 80)
print("2. MIN-MAX SCALING")
print("=" * 80)

minmax_scaler = MinMaxScaler()

scaled_data = minmax_scaler.fit_transform(
    data[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    data[col + "_Scaled"] = scaled_data[:, i]

print("Min-Max Scaling completed.")
print("Values are scaled approximately between 0 and 1.")


# ==========================================================
# 10. L2 NORMALIZATION
# ==========================================================

print("\n" + "=" * 80)
print("3. NORMALIZATION")
print("=" * 80)

normalizer = Normalizer(norm="l2")

normalized_data = normalizer.fit_transform(
    data[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    data[col + "_Normalized"] = normalized_data[:, i]

print("L2 Normalization completed.")
print("Each row is normalized based on its vector magnitude.")


# ==========================================================
# 11. DISPLAY PREPROCESSED DATASET
# ==========================================================

print("\n" + "=" * 80)
print("PREPROCESSED DATASET")
print("=" * 80)

print(data.head())

print("\nDataset Shape:")
print(data.shape)


# ==========================================================
# 12. DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

data.info()

print("\nColumns:")
print(data.columns.tolist())


# ==========================================================
# 13. MISSING VALUES AFTER PREPROCESSING
# ==========================================================

print("\n" + "=" * 80)
print("MISSING VALUES AFTER PREPROCESSING")
print("=" * 80)

print(data.isnull().sum())


# ==========================================================
# 14. DUPLICATE CHECK AFTER PREPROCESSING
# ==========================================================

print("\n" + "=" * 80)
print("DUPLICATE CHECK AFTER PREPROCESSING")
print("=" * 80)

print("Duplicate Records:", data.duplicated().sum())


# ==========================================================
# 15. SAVE PREPROCESSED DATASET
# ==========================================================

data.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("Preprocessed dataset saved successfully.")
print("=" * 80)

print("\nSaved Path:")
print(output_file)


# ==========================================================
# 16. HISTOGRAM OF PREPROCESSED DATASET
# ==========================================================

data.hist(
    figsize=(15, 12),
    bins=10,
    edgecolor="black"
)

plt.suptitle("Histogram of Preprocessed Air Quality Dataset")

plt.tight_layout()

plt.show()
