import pandas as pd
import numpy as np


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

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Check Missing Values Before Cleaning
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

for col in num_cols:
    mean_value = data[col].mean()
    data[col] = data[col].fillna(mean_value)


# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

for col in cat_cols:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)


# ==========================================================
# 7. Target Encoding (Mean Encoding)
# Target Column: PM2.5 (continuous numerical column)
# ==========================================================

target_column = "PM2.5"

target_encoded_df = pd.DataFrame()

for col in cat_cols:

    # Compute mean of PM2.5 grouped by each category
    target_mean_map = data.groupby(col)[target_column].mean()

    # Map encoded values back as Target_{col}
    target_encoded_df["Target_" + col] = data[col].map(
        target_mean_map
    )

target_encoded_df.reset_index(drop=True, inplace=True)

numeric_df = data[num_cols].reset_index(drop=True)

final_output = pd.concat(
    [numeric_df, target_encoded_df],
    axis=1
)


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
    r"\dataset\clean_target_encode_air_quality.csv"
)

final_output.to_csv(
    output_file,
    index=False
)

print("\n======================================")
print("Original Dataset is NOT Modified")
print("Target Encoding Completed Successfully")
print("Target Column Used: PM2.5")
print("Output File:")
print(output_file)
print("======================================")

# Target Encoding:
# The original dataset is cleaned by filling missing values and converting the categorical
# columns wd and station into numerical values by replacing each category with the mean
# PM2.5 value for that category (mean target encoding).
# Encoded columns are stored as Target_wd and Target_station.
