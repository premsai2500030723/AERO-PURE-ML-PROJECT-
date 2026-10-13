"""
load_understand_dataset_M1.py
==============================
Loads and understands the raw Aotizhongxin air quality dataset.
Covers shape, dtypes, missing values, duplicates, statistics and
basic visualisations — same as M1 milestone.

Output
------
output/M1_Load_Understand/

Usage
-----
    python src/load_understand_dataset_M1.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "M1_Load_Understand"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("AIR QUALITY DATASET - LOAD AND UNDERSTAND")
print("Aotizhongxin Station, Beijing")
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
# 3. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 75)
print("FIRST 5 ROWS")
print("=" * 75)

print(df.head())


# ============================================================
# 4. LAST 5 ROWS
# ============================================================

print("\n" + "=" * 75)
print("LAST 5 ROWS")
print("=" * 75)

print(df.tail())


# ============================================================
# 5. DATASET SHAPE
# ============================================================

print("\n" + "=" * 75)
print("DATASET SHAPE")
print("=" * 75)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 6. COLUMN NAMES
# ============================================================

print("\n" + "=" * 75)
print("COLUMN NAMES")
print("=" * 75)

for i, col in enumerate(df.columns, 1):
    print(i, ".", col)


# ============================================================
# 7. DATA TYPES
# ============================================================

print("\n" + "=" * 75)
print("DATA TYPES")
print("=" * 75)

print(df.dtypes)


# ============================================================
# 8. DATASET INFO
# ============================================================

print("\n" + "=" * 75)
print("DATASET INFO")
print("=" * 75)

df.info()


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numerical_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()

print("\n" + "=" * 75)
print("COLUMN TYPES")
print("=" * 75)

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("MISSING VALUES")
print("=" * 75)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": missing.values,
    "Missing_Percentage": missing_pct.values
})

print(missing_report.to_string(index=False))

missing_report.to_csv(
    os.path.join(OUTPUT_FOLDER, "missing_values_report.csv"),
    index=False
)

print("\nMissing values report saved.")


# ============================================================
# 11. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 75)
print("DUPLICATE RECORDS")
print("=" * 75)

print("Duplicate Record Count:", df.duplicated().sum())


# ============================================================
# 12. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STATISTICAL SUMMARY - NUMERICAL COLUMNS")
print("=" * 75)

summary = df[numerical_columns].describe()
print(summary)

summary.to_csv(
    os.path.join(OUTPUT_FOLDER, "statistical_summary.csv")
)

print("\nStatistical summary saved.")


# ============================================================
# 13. UNIQUE VALUES - CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 75)
print("UNIQUE VALUES - CATEGORICAL COLUMNS")
print("=" * 75)

for col in categorical_columns:
    print(f"\n{col} — {df[col].nunique()} unique values:")
    print(df[col].value_counts())


# ============================================================
# 14. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION MATRIX")
print("=" * 75)

corr = df[numerical_columns].corr()
print(corr)

corr.to_csv(
    os.path.join(OUTPUT_FOLDER, "correlation_matrix.csv")
)


# ============================================================
# 15. MISSING VALUES HEATMAP
# ============================================================

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
    os.path.join(OUTPUT_FOLDER, "missing_values_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nMissing values heatmap saved.")


# ============================================================
# 16. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap - Air Quality Dataset")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Correlation heatmap saved.")


# ============================================================
# 17. HISTOGRAMS - NUMERICAL COLUMNS
# ============================================================

for col in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[col].dropna(),
        bins=30,
        kde=True,
        color="skyblue"
    )

    plt.title(f"Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{col}_histogram.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Histograms saved.")


# ============================================================
# 18. BOXPLOTS - NUMERICAL COLUMNS
# ============================================================

for col in numerical_columns:

    plt.figure(figsize=(6, 4))

    sns.boxplot(y=df[col], color="orange")

    plt.title(f"Boxplot - {col}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{col}_boxplot.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Boxplots saved.")


# ============================================================
# 19. PM2.5 MONTHLY TREND
# ============================================================

if "PM2.5" in df.columns and "month" in df.columns:

    monthly_avg = df.groupby("month")["PM2.5"].mean()

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly_avg.index,
        monthly_avg.values,
        marker="o"
    )

    plt.xlabel("Month")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.title("Monthly Average PM2.5 - Aotizhongxin Station")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "PM25_monthly_trend.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Monthly PM2.5 trend saved.")


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print("Dataset          :", INPUT_FILE)
print("Rows             :", df.shape[0])
print("Columns          :", df.shape[1])
print("Numerical Cols   :", len(numerical_columns))
print("Categorical Cols :", len(categorical_columns))
print("Total Missing    :", int(df.isnull().sum().sum()))
print("Duplicates       :", df.duplicated().sum())
print("Output Folder    :", OUTPUT_FOLDER)

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
