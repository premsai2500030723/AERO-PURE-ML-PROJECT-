"""
Correlation_Matrix_heatmap_boxplots_M1.py
==========================================
Computes the correlation matrix for all numeric columns in the
Aotizhongxin air quality dataset and saves:
  - Correlation matrix CSV
  - Correlation heatmap PNG
  - Boxplots for all numeric columns

Output
------
output/Correlation_Matrix_Heatmap_Boxplots_M1/

Usage
-----
    python src/Correlation_Matrix_heatmap_boxplots_M1.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = BASE_DIR / "output" / "Correlation_Matrix_Heatmap_Boxplots_M1"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("CORRELATION MATRIX, HEATMAP AND BOXPLOTS")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 3. SELECT NUMERIC COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)


# ============================================================
# 4. COMPUTE CORRELATION MATRIX
# ============================================================

corr_matrix = df[numeric_cols].corr()

print("\nCorrelation Matrix (first 5 x 5):")
print(corr_matrix.iloc[:5, :5])


# ============================================================
# 5. SAVE CORRELATION MATRIX AS CSV
# ============================================================

csv_path = OUTPUT_FOLDER / "Correlation_Matrix.csv"
corr_matrix.to_csv(csv_path)
print(f"\nCorrelation matrix saved: {csv_path}")


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="white"
)

plt.title("Correlation Heatmap - Air Quality Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()

heatmap_path = OUTPUT_FOLDER / "Correlation_Heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Correlation heatmap saved: {heatmap_path}")


# ============================================================
# 7. CORRELATION WITH PM2.5 (BAR CHART)
# ============================================================

if "PM2.5" in corr_matrix.columns:

    pm25_corr = corr_matrix["PM2.5"].drop("PM2.5").sort_values()

    plt.figure(figsize=(10, 6))

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in pm25_corr]

    pm25_corr.plot(kind="barh", color=colors)

    plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.title("Feature Correlation with PM2.5", fontsize=13, fontweight="bold")
    plt.tight_layout()

    bar_path = OUTPUT_FOLDER / "PM25_Correlation_Bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"PM2.5 correlation bar chart saved: {bar_path}")


# ============================================================
# 8. BOXPLOTS FOR ALL NUMERIC COLUMNS
# ============================================================

print("\nGenerating boxplots …")

for col in numeric_cols:

    plt.figure(figsize=(6, 4))

    sns.boxplot(y=df[col], color="skyblue")

    plt.title(f"Boxplot — {col}", fontsize=12)
    plt.tight_layout()

    boxplot_path = OUTPUT_FOLDER / f"{col}_Boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()

print("All boxplots saved.")


# ============================================================
# 9. COMBINED BOXPLOT (KEY POLLUTANTS)
# ============================================================

pollutants = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if c in df.columns]

if pollutants:

    fig, axes = plt.subplots(1, len(pollutants), figsize=(16, 5))

    for ax, col in zip(axes, pollutants):
        sns.boxplot(y=df[col], ax=ax, color="lightcoral")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")

    fig.suptitle("Key Pollutants — Boxplot Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    combined_path = OUTPUT_FOLDER / "Pollutants_Combined_Boxplot.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined pollutant boxplot saved: {combined_path}")


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("COMPLETED SUCCESSFULLY")
print("=" * 75)
print(f"\nOutput folder: {OUTPUT_FOLDER}")
print("\nOriginal dataset was NOT modified.")
