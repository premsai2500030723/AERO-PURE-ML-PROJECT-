import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = r"C:\Users\pream\PycharmProjects\ML- project\outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("FIRST FIVE RECORDS")
print("=" * 60)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# =====================================================
# NUMERIC & CATEGORICAL COLUMNS
# =====================================================

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# =====================================================
# STATISTICAL SUMMARY
# =====================================================

summary = df.describe(include="all")

print("\nStatistical Summary:")
print(summary)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Statistical_Summary.csv"
    )
)


# =====================================================
# MISSING VALUES REPORT
# =====================================================

missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage":
        (df.isnull().sum().values / len(df)) * 100
})

missing_values.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Missing_Values_Report.csv"
    ),
    index=False
)


# =====================================================
# CORRELATION MATRIX
# =====================================================

numeric_df = df[numeric_cols]

corr_matrix = numeric_df.corr()

corr_matrix.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Matrix.csv"
    )
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Heatmap.png"
    ),
    dpi=300
)

plt.close()


# =====================================================
# HISTOGRAMS
# =====================================================

for col in numeric_cols:

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
        os.path.join(
            OUTPUT_FOLDER,
            f"{col}_Histogram.png"
        ),
        dpi=300
    )

    plt.close()


# =====================================================
# BOXPLOTS
# =====================================================

for col in numeric_cols:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        y=df[col],
        color="orange"
    )

    plt.title(f"Boxplot - {col}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{col}_Boxplot.png"
        ),
        dpi=300
    )

    plt.close()


# =====================================================
# OUTLIER DETECTION
# =====================================================

for col in numeric_cols:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        y=df[col],
        color="red"
    )

    plt.title(f"Outlier Detection - {col}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{col}_Outlier.png"
        ),
        dpi=300
    )

    plt.close()


# =====================================================
# COUNT PLOTS FOR CATEGORICAL COLUMNS
# =====================================================

for col in ["wd", "station"]:

    if col in df.columns:

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=df,
            x=col
        )

        plt.xticks(rotation=90)

        plt.title(f"Count Plot - {col}")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                f"{col}_CountPlot.png"
            ),
            dpi=300
        )

        plt.close()


# =====================================================
# PAIR PLOT
# =====================================================

selected_cols = [
    col for col in
    ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    if col in df.columns
]

if len(selected_cols) > 1:

    pairplot = sns.pairplot(
        df[selected_cols].dropna()
    )

    pairplot.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Pollution_PairPlot.png"
        )
    )

    plt.close()


# =====================================================
# MISSING VALUE HEATMAP
# =====================================================

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis",
    yticklabels=False
)

plt.title("Missing Values Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Missing_Values_Heatmap.png"
    ),
    dpi=300
)

plt.close()


# =====================================================
# SCATTER PLOT: year vs PM2.5
# =====================================================

if "year" in df.columns and "PM2.5" in df.columns:

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="year",
        y="PM2.5",
        alpha=0.5
    )

    plt.title("Year vs PM2.5")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Year_vs_PM25_Scatter.png"
        ),
        dpi=300
    )

    plt.close()


print("\n" + "=" * 60)
print("EDA Completed Successfully.")
print("=" * 60)
print(f"Output Folder: {OUTPUT_FOLDER}")
