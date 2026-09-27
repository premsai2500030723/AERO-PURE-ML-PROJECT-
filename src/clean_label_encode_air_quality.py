"""
clean_label_encode_air_quality.py
==================================
Loads the raw Aotizhongxin air quality dataset, performs cleaning
(deduplication, missing-value imputation) and applies **Label Encoding**
to all categorical columns (wd, station).

Output
------
dataset/clean_label_encode_air_quality.csv

Usage
-----
    python src/clean_label_encode_air_quality.py

Notes
-----
- The original raw CSV is never modified; a copy is used throughout.
- LabelEncoder assigns an integer rank to each unique category value.
- Fitting is done per-column; encoders are stored in label_encoders dict.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Load Air Quality Dataset
# Original dataset will NOT be modified
# ==========================================================

df = pd.read_csv(
    r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"
)

# Create a copy for processing
data = df.copy()


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values (Before)
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before = data.shape[0]

data = data.drop_duplicates()

after = data.shape[0]

print("\nDuplicate Records Removed:", before - after)


# ==========================================================
# 4. Identify Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = data.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================

if len(num_cols) > 0:

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    data[num_cols] = num_imputer.fit_transform(
        data[num_cols]
    )


# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

if len(cat_cols) > 0:

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


# ==========================================================
# 7. Label Encoding
# ==========================================================

label_encoders = {}

for col in cat_cols:

    encoder = LabelEncoder()

    data[col] = encoder.fit_transform(
        data[col]
    )

    label_encoders[col] = encoder


# ==========================================================
# 8. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 9. Save Result
# ==========================================================

output_file = (
    r"C:\Users\pream\PycharmProjects\ML- project"
    r"\dataset\clean_label_encode_air_quality.csv"
)

data.to_csv(
    output_file,
    index=False
)

print("\n======================================")
print("Original Dataset is NOT Modified")
print("Label Encoding Completed Successfully")
print("Output File:")
print(output_file)
print("======================================")
