"""
Linear_Regression_CENE_GD_Compare_M2.py
=========================================
Compares Closed-Form Normal Equation vs Gradient Descent
for Linear Regression on the Air Quality Dataset.
Target: PM2.5 prediction.

Output
------
output/Linear_Regression_CFNE_GD_Compare_M2/

Usage
-----
    python src/Linear_Regression_CENE_GD_Compare_M2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "Linear_Regression_CFNE_GD_Compare_M2"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Output folder:")
print(OUTPUT_FOLDER)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 75)
print("LINEAR REGRESSION - NORMAL EQUATION vs GRADIENT DESCENT")
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
# 4. CONVERT ALL COLUMNS TO NUMERIC
# ============================================================

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ============================================================
# 5. REPLACE INFINITE VALUES WITH NaN
# ============================================================

data = data.replace([np.inf, -np.inf], np.nan)

print("\nNaN values after numeric conversion:")
nan_counts = data.isna().sum()
print(nan_counts[nan_counts > 0])


# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

# Features: all columns except PM2.5
# Target  : PM2.5

target_column = "PM2.5"

feature_columns = [
    col for col in data.columns
    if col != target_column
]

X_df     = data[feature_columns].copy()
y_series = data[target_column].copy()


# ============================================================
# 7. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

valid_target      = y_series.notna()
removed_rows      = (~valid_target).sum()

if removed_rows > 0:
    print(f"\nRemoving {removed_rows} rows with missing target.")
    X_df     = X_df.loc[valid_target].reset_index(drop=True)
    y_series = y_series.loc[valid_target].reset_index(drop=True)


# ============================================================
# 8. FILL MISSING FEATURE VALUES WITH MEDIAN
# ============================================================

print(
    "\nMissing feature values before imputation:",
    X_df.isna().sum().sum()
)

for col in X_df.columns:
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
# 9. CONVERT TO NUMPY AND REMOVE INVALID ROWS
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

print("\nFinal NaN/Inf check:")
print("NaN in X:", np.isnan(X).sum())
print("NaN in y:", np.isnan(y).sum())
print("Inf in X:", np.isinf(X).sum())
print("Inf in y:", np.isinf(y).sum())


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
# 12. CLOSED-FORM NORMAL EQUATION
# ============================================================

print("\nRunning Closed-Form Normal Equation...")

# Add bias column
X_train_bias = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_bias = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]

# theta = pinv(X) * y
theta = np.linalg.pinv(X_train_bias).dot(y_train)


# ============================================================
# 13. NORMAL EQUATION PREDICTION
# ============================================================

pred_normal = X_test_bias.dot(theta)

print(
    "NaN in Normal Equation predictions:",
    np.isnan(pred_normal).sum()
)

print(
    "Inf in Normal Equation predictions:",
    np.isinf(pred_normal).sum()
)


# ============================================================
# 14. NORMAL EQUATION METRICS
# ============================================================

mse_normal = mean_squared_error(y_test, pred_normal)
r2_normal  = r2_score(y_test, pred_normal)

print("\n------ Closed-Form Normal Equation ------")
print("MSE     :", round(mse_normal, 4))
print("R2 Score:", round(r2_normal, 4))


# ============================================================
# 15. GRADIENT DESCENT
# ============================================================

print("\nRunning Gradient Descent...")

X_train_gd = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]

m             = len(y_train)
theta_gd      = np.zeros(X_train_gd.shape[1])
learning_rate = 0.01
epochs        = 1000
loss_history  = []


# ============================================================
# 16. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    predictions = X_train_gd.dot(theta_gd)

    errors      = predictions - y_train

    gradients   = (2 / m) * X_train_gd.T.dot(errors)

    theta_gd   -= learning_rate * gradients

    current_predictions = X_train_gd.dot(theta_gd)

    loss = np.mean((current_predictions - y_train) ** 2)

    loss_history.append(loss)


# ============================================================
# 17. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = X_test_gd.dot(theta_gd)

print(
    "NaN in Gradient Descent predictions:",
    np.isnan(pred_gd).sum()
)

print(
    "Inf in Gradient Descent predictions:",
    np.isinf(pred_gd).sum()
)


# ============================================================
# 18. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(y_test, pred_gd)
r2_gd  = r2_score(y_test, pred_gd)

print("\n------ Gradient Descent ------")
print("MSE     :", round(mse_gd, 4))
print("R2 Score:", round(r2_gd, 4))


# ============================================================
# 19. COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("COMPARISON")
print("=" * 75)

print("\nNormal Equation")
print("MSE :", mse_normal)
print("R2  :", r2_normal)

print("\nGradient Descent")
print("MSE :", mse_gd)
print("R2  :", r2_gd)

comparison_df = pd.DataFrame({
    "Method": ["Normal Equation", "Gradient Descent"],
    "MSE":    [mse_normal, mse_gd],
    "R2":     [r2_normal,  r2_gd]
})

comparison_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "comparison_metrics.csv"),
    index=False
)

print("\nComparison metrics saved.")


# ============================================================
# 20. IMAGE 1 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, pred_normal, alpha=0.3, s=5, label="Normal Equation")
plt.scatter(y_test, pred_gd,     alpha=0.3, s=5, label="Gradient Descent")

min_val = min(y_test.min(), pred_normal.min(), pred_gd.min())
max_val = max(y_test.max(), pred_normal.max(), pred_gd.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 (Normal Eq. vs GD)")
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
# 21. IMAGE 2 - RESIDUAL COMPARISON
# ============================================================

normal_residuals = y_test - pred_normal
gd_residuals     = y_test - pred_gd

plt.figure(figsize=(9, 6))

plt.scatter(pred_normal, normal_residuals, alpha=0.3, s=5, label="Normal Equation")
plt.scatter(pred_gd,     gd_residuals,     alpha=0.3, s=5, label="Gradient Descent")
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted PM2.5")
plt.ylabel("Residuals")
plt.title("Residual Comparison - Normal Eq. vs Gradient Descent")
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
# 22. IMAGE 3 - GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(range(1, epochs + 1), loss_history)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Convergence - PM2.5 Prediction")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "gradient_descent_loss.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: gradient_descent_loss.png")


# ============================================================
# 23. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)

print("NaN in y_test      :", np.isnan(y_test).sum())
print("NaN in pred_normal :", np.isnan(pred_normal).sum())
print("NaN in pred_gd     :", np.isnan(pred_gd).sum())
print("Inf in y_test      :", np.isinf(y_test).sum())
print("Inf in pred_normal :", np.isinf(pred_normal).sum())
print("Inf in pred_gd     :", np.isinf(pred_gd).sum())


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\n1. actual_vs_predicted.png")
print("2. residual_comparison.png")
print("3. gradient_descent_loss.png")
print("4. comparison_metrics.csv")

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
