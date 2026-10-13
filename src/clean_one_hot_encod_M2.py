"""
clean_one_hot_encoding_air_quality.py
=======================================
Loads the raw Aotizhongxin air quality dataset, performs cleaning
(deduplication, missing-value imputation) and applies **One-Hot Encoding**
to all categorical columns (wd, station).

Output
------
dataset/clean_one_hot_encoding_air_quality.csv

Usage
-----
    python src/clean_one_hot_encoding_air_quality.py

Notes
-----
- The original raw CSV is never modified; a copy is used throughout.
- OneHotEncoder expands each categorical column into binary indicator columns.
- Encoded feature names follow the pattern: <original_col>_<category_value>.
- Resulting dataset has more columns than the raw input (wd has 16+ categories).
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
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

for col in data.select_dtypes(include=["object", "str"]).columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before_duplicates = data.shape[0]

data = data.drop_duplicates()

after_duplicates = data.shape[0]

print("\nDuplicate Records Removed:",
      before_duplicates - after_duplicates)


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
# 7. One-Hot Encoding
# ==========================================================

if len(cat_cols) > 0:

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded_values = encoder.fit_transform(
        data[cat_cols]
    )

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Reset index
    encoded_df.reset_index(
        drop=True,
        inplace=True
    )

    # Keep numerical columns
    numeric_df = data[num_cols].reset_index(
        drop=True
    )

    # Combine numerical and encoded columns
    final_output = pd.concat(
        [
            numeric_df,
            encoded_df
        ],
        axis=1
    )

else:

    final_output = data.copy()


# ==========================================================
# 8. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())


# ==========================================================
# 9. Save Final Result
# ==========================================================

output_file = (
    r"C:\Users\pream\PycharmProjects\ML- project"
    r"\dataset\clean_one_hot_encoding_air_quality.csv"
)

final_output.to_csv(
    output_file,
    index=False
)


print("\n======================================")
print("Original Dataset is NOT Modified")
print("One-Hot Encoding Completed Successfully ?")
print("Output File:")
print(output_file)
print("======================================")
#The original dataset is cleaned by filling missing values and converting the categorical
# columns wd and station into multiple binary (0/1) columns using One-Hot Encoding.

# reviewed: one-hot encoding v1.1
