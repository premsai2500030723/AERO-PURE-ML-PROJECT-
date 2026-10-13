"""
clean_del_mean_model_missing_imputer_M2.py
===========================================
Loads the raw Aotizhongxin air quality dataset and demonstrates
three missing value handling strategies:

  1. DELETION     — drop rows that contain any missing value
  2. MEAN IMPUTATION  — fill missing values with column mean
  3. MODEL IMPUTATION — fill missing values using KNNImputer

Output
------
dataset/clean_del_air_quality.csv          (deletion method)
dataset/clean_mean_air_quality.csv         (mean imputation)
dataset/clean_model_imputed_air_quality.csv (KNN imputation)

Usage
-----
    python src/clean_del_mean_model_missing_imputer_M2.py
"""

import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = r"C:\Users\pream\PycharmProjects\ML- project\dataset"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("MISSING VALUE HANDLING - DELETE / MEAN / MODEL IMPUTATION")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nDataset not found.\nCheck this path:\n" + INPUT_FILE
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 3. IDENTIFY MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("MISSING VALUES BEFORE HANDLING")
print("=" * 75)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Column"             : df.columns,
    "Missing_Count"      : missing.values,
    "Missing_Percentage" : missing_pct.values
})

print(missing_report[missing_report["Missing_Count"] > 0].to_string(index=False))

print("\nTotal missing values:", int(df.isnull().sum().sum()))


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)


# ============================================================
# 5. METHOD 1 — DELETION
# ============================================================

print("\n" + "=" * 75)
print("METHOD 1 — DELETION (Drop rows with any missing value)")
print("=" * 75)

data_del = df.copy()

rows_before = data_del.shape[0]

data_del = data_del.dropna()

rows_after = data_del.shape[0]

rows_removed = rows_before - rows_after

print(f"Rows before deletion : {rows_before}")
print(f"Rows after deletion  : {rows_after}")
print(f"Rows removed         : {rows_removed}")
print(f"Data lost            : {round(rows_removed / rows_before * 100, 2)}%")

print("\nMissing values after deletion:")
print(data_del.isnull().sum().sum())

output_del = os.path.join(OUTPUT_FOLDER, "clean_del_air_quality.csv")

data_del.to_csv(output_del, index=False)

print("\nDeletion result saved:")
print(output_del)


# ============================================================
# 6. METHOD 2 — MEAN IMPUTATION
# ============================================================

print("\n" + "=" * 75)
print("METHOD 2 — MEAN IMPUTATION")
print("=" * 75)

data_mean = df.copy()

# Strip spaces from categorical columns
for col in cat_cols:
    data_mean[col] = data_mean[col].astype(str).str.strip()

# Remove duplicates
data_mean = data_mean.drop_duplicates()

# Fill numerical missing values with mean
num_imputer = SimpleImputer(strategy="mean")
data_mean[num_cols] = num_imputer.fit_transform(data_mean[num_cols])

# Fill categorical missing values with mode
cat_imputer = SimpleImputer(strategy="most_frequent")
data_mean[cat_cols] = cat_imputer.fit_transform(data_mean[cat_cols])

print("Mean imputation completed.")
print("Missing values after mean imputation:")
print(data_mean.isnull().sum().sum())

output_mean = os.path.join(OUTPUT_FOLDER, "clean_mean_air_quality.csv")

data_mean.to_csv(output_mean, index=False)

print("\nMean imputation result saved:")
print(output_mean)


# ============================================================
# 7. METHOD 3 — MODEL IMPUTATION (KNNImputer)
# ============================================================

print("\n" + "=" * 75)
print("METHOD 3 — MODEL IMPUTATION (KNNImputer, k=5)")
print("=" * 75)

data_model = df.copy()

# Strip spaces from categorical columns
for col in cat_cols:
    data_model[col] = data_model[col].astype(str).str.strip()

# Remove duplicates
data_model = data_model.drop_duplicates()

# Fill categorical columns with mode first
cat_imputer = SimpleImputer(strategy="most_frequent")
data_model[cat_cols] = cat_imputer.fit_transform(data_model[cat_cols])

# Apply KNNImputer only on numerical columns
knn_imputer = KNNImputer(n_neighbors=5)

data_model[num_cols] = knn_imputer.fit_transform(data_model[num_cols])

print("KNN model imputation completed.")
print("Missing values after KNN imputation:")
print(data_model.isnull().sum().sum())

output_model = os.path.join(OUTPUT_FOLDER, "clean_model_imputed_air_quality.csv")

data_model.to_csv(output_model, index=False)

print("\nKNN imputation result saved:")
print(output_model)


# ============================================================
# 8. COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("COMPARISON SUMMARY")
print("=" * 75)

print(f"Original dataset rows       : {df.shape[0]}")
print(f"After Deletion rows         : {data_del.shape[0]}")
print(f"After Mean Imputation rows  : {data_mean.shape[0]}")
print(f"After KNN Imputation rows   : {data_model.shape[0]}")

print("\nMissing values remaining:")
print(f"After Deletion              : {int(data_del.isnull().sum().sum())}")
print(f"After Mean Imputation       : {int(data_mean.isnull().sum().sum())}")
print(f"After KNN Imputation        : {int(data_model.isnull().sum().sum())}")


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\n1. clean_del_air_quality.csv          (Deletion method)")
print("2. clean_mean_air_quality.csv         (Mean imputation)")
print("3. clean_model_imputed_air_quality.csv (KNN imputation)")

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
