"""
clean_ordinal_encode_air_quality.py
=====================================
Loads the raw Aotizhongxin air quality dataset, performs cleaning
(deduplication, missing-value imputation) and applies **Ordinal Encoding**
to all categorical columns (wd, station).

Output
------
dataset/clean_ordinal_encode_air_quality.csv

Usage
-----
    python src/clean_ordinal_encode_air_quality.py

Notes
-----
- The original raw CSV is never modified; a copy is used throughout.
- OrdinalEncoder assigns integer ranks based on lexicographic order by default.
- Encoded columns are renamed with the prefix 'Ordinal_' for clarity.
- Best suited for tree-based models that can handle ordinal relationships.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
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
# 7. Ordinal Encoding
# ==========================================================

encoder = OrdinalEncoder()

encoded_values = encoder.fit_transform(
    data[cat_cols]
)

# Create ordinal encoded DataFrame with named columns
ordinal_df = pd.DataFrame(
    encoded_values,
    columns=["Ordinal_" + col for col in cat_cols]
)

ordinal_df.reset_index(drop=True, inplace=True)


# ==========================================================
# 8. Merge Numerical + Ordinal Encoded Columns
# ==========================================================

numeric_df = data[num_cols].reset_index(drop=True)

final_output = pd.concat(
    [
        numeric_df,
        ordinal_df
    ],
    axis=1
)


# ==========================================================
# 9. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())


# ==========================================================
# 10. Save Final Result
# ==========================================================

output_file = (
    r"C:\Users\pream\PycharmProjects\ML- project"
    r"\dataset\clean_ordinal_encode_air_quality.csv"
)

final_output.to_csv(
    output_file,
    index=False
)

print("\n======================================")
print("Original Dataset is NOT Modified")
print("Ordinal Encoding Completed Successfully")
print("Output File:")
print(output_file)
print("======================================")
