import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# FILE PATHS
# ==========================================================

input_file = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

output_folder = r"C:\Users\pream\PycharmProjects\ML- project\outputs"

os.makedirs(output_folder, exist_ok=True)


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv(input_file)

print("=" * 60)
print("AIR QUALITY DATASET")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nFirst 6 Columns:")
print(df.iloc[:, 0:6])

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)


# ==========================================================
# 2. MISSING VALUES
# ==========================================================

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)

print(df.isnull().sum())


# ==========================================================
# 3. DUPLICATE RECORDS
# ==========================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Duplicate Record Count:", df.duplicated().sum())


# ==========================================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================================

num_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n" + "=" * 60)
print("COLUMN TYPES")
print("=" * 60)

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ==========================================================
# 5. MISSING VALUES HEATMAP
# ==========================================================

plt.figure(figsize=(14, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis",
    yticklabels=False
)

plt.title("Missing Values Heatmap - Air Quality Dataset")
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Missing_Values_Heatmap.png"
    ),
    dpi=300
)

plt.show()

print("\nMissing Values Heatmap saved to outputs folder.")

print("\n" + "=" * 60)
print("DATASET LOADING COMPLETED SUCCESSFULLY")
print("=" * 60)
