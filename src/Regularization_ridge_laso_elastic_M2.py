"""
Regularization_ridge_laso_elastic_M2.py
=========================================
Ridge, Lasso and ElasticNet Regularization on Air Quality Dataset.
Predicts PM2.5 using pollution and weather features.
Compares all three regularization methods with metrics and plots.

Output
------
output/Regularization_Ridge_Lasso_ElasticNet_M2/

Usage
-----
    python src/Regularization_ridge_laso_elastic_M2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
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
    "Regularization_Ridge_Lasso_ElasticNet_M2"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("REGULARIZATION - RIDGE, LASSO, ELASTICNET")
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
# 3. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

total_nan = data.isna().sum().sum()
numeric_data = data.select_dtypes(include=[np.number])
total_inf = np.isinf(numeric_data.to_numpy()).sum()

print("Total NaN values     :", total_nan)
print("Total infinite values:", total_inf)


# ============================================================
# 4. CONVERT TO NUMERIC
# ============================================================

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ============================================================
# 5. REPLACE INFINITE VALUES
# ============================================================

data = data.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

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

X_df     = data[feature_columns].copy()
y_series = data[target_column].copy()


# ============================================================
# 7. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

valid_target = y_series.notna()
removed      = (~valid_target).sum()

if removed > 0:
    print(f"\nRemoving {removed} rows with missing target.")
    X_df     = X_df.loc[valid_target].reset_index(drop=True)
    y_series = y_series.loc[valid_target].reset_index(drop=True)


# ============================================================
# 8. FILL MISSING FEATURE VALUES WITH MEDIAN
# ============================================================

print(
    "\nMissing feature values before imputation:",
    X_df.isna().sum().sum()
)

for col in feature_columns:
    if X_df[col].isna().any():
        median_val = X_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_df[col] = X_df[col].fillna(median_val)

print(
    "Missing feature values after imputation:",
    X_df.isna().sum().sum()
)


# ============================================================
# 9. CONVERT TO NUMPY
# ============================================================

X_array = X_df.to_numpy(dtype=float)
y_array = y_series.to_numpy(dtype=float)

valid_rows = (
    np.isfinite(X_array).all(axis=1) & np.isfinite(y_array)
)

invalid = (~valid_rows).sum()

if invalid > 0:
    print(f"\nRemoving {invalid} remaining invalid rows.")
    X_array = X_array[valid_rows]
    y_array = y_array[valid_rows]

X = X_array
y = y_array

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
# 12. TRAIN RIDGE REGRESSION
# ============================================================

print("\nTraining Ridge Regression (alpha=1.0)...")

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

pred_ridge = ridge_model.predict(X_test_scaled)

print("Ridge training completed.")


# ============================================================
# 13. TRAIN LASSO REGRESSION
# ============================================================

print("\nTraining Lasso Regression (alpha=0.1)...")

lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

pred_lasso = lasso_model.predict(X_test_scaled)

print("Lasso training completed.")


# ============================================================
# 14. TRAIN ELASTICNET REGRESSION
# ============================================================

print("\nTraining ElasticNet Regression (alpha=0.1, l1_ratio=0.5)...")

elastic_model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    random_state=42,
    max_iter=10000
)

elastic_model.fit(X_train_scaled, y_train)

pred_elastic = elastic_model.predict(X_test_scaled)

print("ElasticNet training completed.")


# ============================================================
# 15. METRICS FUNCTION
# ============================================================

def compute_metrics(y_true, y_pred, name):

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"R2    : {r2:.4f}")

    return {
        "Method": name,
        "MSE":    round(mse, 4),
        "RMSE":   round(rmse, 4),
        "MAE":    round(mae, 4),
        "R2":     round(r2, 4)
    }


# ============================================================
# 16. PRINT METRICS
# ============================================================

print("\n" + "=" * 75)
print("MODEL EVALUATION METRICS")
print("=" * 75)

ridge_metrics   = compute_metrics(y_test, pred_ridge,   "Ridge")
lasso_metrics   = compute_metrics(y_test, pred_lasso,   "Lasso")
elastic_metrics = compute_metrics(y_test, pred_elastic, "ElasticNet")


# ============================================================
# 17. SAVE COMPARISON METRICS
# ============================================================

comparison_df = pd.DataFrame([
    ridge_metrics,
    lasso_metrics,
    elastic_metrics
])

comparison_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "comparison_metrics.csv"),
    index=False
)

print("\nComparison metrics saved.")


# ============================================================
# 18. IMAGE 1 - ACTUAL VS PREDICTED (ALL 3 METHODS)
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(y_test, pred_ridge,   alpha=0.3, s=5, label="Ridge")
plt.scatter(y_test, pred_lasso,   alpha=0.3, s=5, label="Lasso")
plt.scatter(y_test, pred_elastic, alpha=0.3, s=5, label="ElasticNet")

min_val = y_test.min()
max_val = y_test.max()

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="black",
    label="Perfect Prediction"
)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 - Ridge, Lasso, ElasticNet")
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
# 19. IMAGE 2 - RESIDUAL COMPARISON
# ============================================================

ridge_residuals   = y_test - pred_ridge
lasso_residuals   = y_test - pred_lasso
elastic_residuals = y_test - pred_elastic

plt.figure(figsize=(10, 6))

plt.scatter(pred_ridge,   ridge_residuals,   alpha=0.3, s=5, label="Ridge")
plt.scatter(pred_lasso,   lasso_residuals,   alpha=0.3, s=5, label="Lasso")
plt.scatter(pred_elastic, elastic_residuals, alpha=0.3, s=5, label="ElasticNet")

plt.axhline(y=0, linestyle="--", color="black")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residuals")
plt.title("Residual Comparison - Ridge, Lasso, ElasticNet")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "residual_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: residual_comparison.png")


# ============================================================
# 20. IMAGE 3 - COEFFICIENT COMPARISON
# ============================================================

x_pos = np.arange(len(feature_columns))
width = 0.25

plt.figure(figsize=(13, 6))

plt.bar(x_pos - width, ridge_model.coef_,   width, label="Ridge")
plt.bar(x_pos,         lasso_model.coef_,   width, label="Lasso")
plt.bar(x_pos + width, elastic_model.coef_, width, label="ElasticNet")

plt.axhline(y=0, linestyle="--", color="black")
plt.xticks(x_pos, feature_columns, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.title("Coefficient Comparison - Ridge, Lasso, ElasticNet")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "coefficient_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: coefficient_comparison.png")


# ============================================================
# 21. IMAGE 4 - METRICS BAR CHART
# ============================================================

metrics_names  = ["MSE", "RMSE", "MAE", "R2"]
ridge_vals     = [ridge_metrics["MSE"],   ridge_metrics["RMSE"],   ridge_metrics["MAE"],   ridge_metrics["R2"]]
lasso_vals     = [lasso_metrics["MSE"],   lasso_metrics["RMSE"],   lasso_metrics["MAE"],   lasso_metrics["R2"]]
elastic_vals   = [elastic_metrics["MSE"], elastic_metrics["RMSE"], elastic_metrics["MAE"], elastic_metrics["R2"]]

x_pos = np.arange(len(metrics_names))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x_pos - width, ridge_vals,   width, label="Ridge")
plt.bar(x_pos,         lasso_vals,   width, label="Lasso")
plt.bar(x_pos + width, elastic_vals, width, label="ElasticNet")

plt.xticks(x_pos, metrics_names)
plt.ylabel("Score")
plt.title("Metrics Comparison - Ridge, Lasso, ElasticNet")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "metrics_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: metrics_comparison.png")


# ============================================================
# 22. LASSO ZERO COEFFICIENTS
# ============================================================

print("\n" + "=" * 75)
print("LASSO - ZERO COEFFICIENTS (FEATURE SELECTION)")
print("=" * 75)

for col, coef in zip(feature_columns, lasso_model.coef_):
    status = "ZEROED OUT" if coef == 0.0 else f"{coef:.4f}"
    print(f"{col:<12}: {status}")


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\n1. actual_vs_predicted.png")
print("2. residual_comparison.png")
print("3. coefficient_comparison.png")
print("4. metrics_comparison.png")
print("5. comparison_metrics.csv")

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
