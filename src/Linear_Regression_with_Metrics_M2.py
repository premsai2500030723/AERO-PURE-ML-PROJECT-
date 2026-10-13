"""
Linear_Regression_with_Metrics_M2.py
======================================
Linear Regression on Air Quality Dataset.
Predicts PM2.5 using pollution and weather features.
Includes full evaluation metrics and visualisation.

Output
------
output/Linear_Regression_with_Metrics_M2/

Usage
-----
    python src/Linear_Regression_with_Metrics_M2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "Linear_Regression_with_Metrics_M2"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("LINEAR REGRESSION WITH METRICS")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nDataset not found.\nCheck this path:\n" + INPUT_FILE
    )

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

# Features: key pollution + weather columns
# Target  : PM2.5

feature_columns = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

target_column = "PM2.5"

print("\nFeature columns:", feature_columns)
print("Target column  :", target_column)


# ============================================================
# 4. SELECT X AND y
# ============================================================

X_df = data[feature_columns].copy()
y_series = data[target_column].copy()


# ============================================================
# 5. CONVERT TO NUMERIC
# ============================================================

for col in feature_columns:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

y_series = pd.to_numeric(y_series, errors="coerce")


# ============================================================
# 6. REPLACE INFINITE VALUES
# ============================================================

X_df = X_df.replace([np.inf, -np.inf], np.nan)
y_series = y_series.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 7. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

valid_target = y_series.notna()
removed = (~valid_target).sum()

if removed > 0:
    print(f"\nRemoving {removed} rows with missing target.")
    X_df     = X_df.loc[valid_target].reset_index(drop=True)
    y_series = y_series.loc[valid_target].reset_index(drop=True)


# ============================================================
# 8. FILL MISSING FEATURE VALUES WITH MEDIAN
# ============================================================

print("\nMissing feature values before imputation:")
print(X_df.isnull().sum())

for col in feature_columns:
    if X_df[col].isnull().any():
        median_val = X_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_df[col] = X_df[col].fillna(median_val)

print("\nMissing feature values after imputation:")
print(X_df.isnull().sum())


# ============================================================
# 9. CONVERT TO NUMPY
# ============================================================

X = X_df.to_numpy(dtype=float)
y = y_series.to_numpy(dtype=float)

# Remove any remaining invalid rows
valid_rows = (
    np.isfinite(X).all(axis=1) & np.isfinite(y)
)

X = X[valid_rows]
y = y[valid_rows]

print("\nFinal dataset shape:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 11. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 12. TRAIN LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression model...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Training completed.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)


# ============================================================
# 14. METRICS - TRAIN
# ============================================================

train_mse  = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_mae  = mean_absolute_error(y_train, y_pred_train)
train_r2   = r2_score(y_train, y_pred_train)


# ============================================================
# 15. METRICS - TEST
# ============================================================

test_mse  = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae  = mean_absolute_error(y_test, y_pred_test)
test_r2   = r2_score(y_test, y_pred_test)


# ============================================================
# 16. PRINT METRICS
# ============================================================

print("\n" + "=" * 75)
print("MODEL EVALUATION METRICS")
print("=" * 75)

print("\nTRAIN SET")
print("-" * 40)
print(f"MSE   : {train_mse:.4f}")
print(f"RMSE  : {train_rmse:.4f}")
print(f"MAE   : {train_mae:.4f}")
print(f"R2    : {train_r2:.4f}")

print("\nTEST SET")
print("-" * 40)
print(f"MSE   : {test_mse:.4f}")
print(f"RMSE  : {test_rmse:.4f}")
print(f"MAE   : {test_mae:.4f}")
print(f"R2    : {test_r2:.4f}")


# ============================================================
# 17. COEFFICIENTS
# ============================================================

print("\n" + "=" * 75)
print("MODEL COEFFICIENTS")
print("=" * 75)

print(f"Intercept: {model.intercept_:.4f}")

for col, coef in zip(feature_columns, model.coef_):
    print(f"{col:<10}: {coef:.4f}")


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE", "R2_Score"],
    "Train":  [train_mse, train_rmse, train_mae, train_r2],
    "Test":   [test_mse,  test_rmse,  test_mae,  test_r2]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "metrics.csv"),
    index=False
)

print("\nMetrics saved.")


# ============================================================
# 19. IMAGE 1 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_test,
    alpha=0.3,
    s=5,
    color="steelblue"
)

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 - Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "actual_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: actual_vs_predicted.png")


# ============================================================
# 20. IMAGE 2 - RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred_test

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred_test,
    residuals,
    alpha=0.3,
    s=5,
    color="darkorange"
)

plt.axhline(y=0, linestyle="--", color="red")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "residual_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: residual_plot.png")


# ============================================================
# 21. IMAGE 3 - FEATURE COEFFICIENTS
# ============================================================

plt.figure(figsize=(10, 6))

colors = [
    "green" if c >= 0 else "red"
    for c in model.coef_
]

plt.barh(feature_columns, model.coef_, color=colors)
plt.axvline(x=0, linestyle="--", color="black")
plt.xlabel("Coefficient Value")
plt.title("Feature Coefficients - Linear Regression")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "feature_coefficients.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: feature_coefficients.png")


# ============================================================
# 22. IMAGE 4 - RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(residuals, bins=50, color="steelblue", edgecolor="black")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution - Linear Regression")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "residual_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: residual_distribution.png")


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
