"""
Logistic_Regression_Binary_Classifi_M2.py
===========================================
Binary Classification using Logistic Regression on Air Quality Dataset.
Creates a binary target: PM2.5 >= 75 µg/m³ → Unhealthy (1), else Healthy (0).
Includes full evaluation metrics and visualisation.

Output
------
output/Logistic_Regression_Binary_Classifi_M2/

Usage
-----
    python src/Logistic_Regression_Binary_Classifi_M2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "Logistic_Regression_Binary_Classifi_M2"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("LOGISTIC REGRESSION - BINARY CLASSIFICATION")
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
# 3. CREATE BINARY TARGET
# ============================================================

# PM2.5 >= 75 µg/m³  →  Unhealthy (1)
# PM2.5 <  75 µg/m³  →  Healthy   (0)

THRESHOLD = 75.0

data = data.dropna(subset=["PM2.5"])

data["AQ_Label"] = (data["PM2.5"] >= THRESHOLD).astype(int)

print(f"\nBinary target created (threshold = {THRESHOLD} µg/m³):")
print(data["AQ_Label"].value_counts())
print("0 = Healthy (PM2.5 < 75)")
print("1 = Unhealthy (PM2.5 >= 75)")


# ============================================================
# 4. FEATURE COLUMNS
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

target_column = "AQ_Label"

X_df     = data[feature_columns].copy()
y_series = data[target_column].copy()


# ============================================================
# 5. CONVERT TO NUMERIC
# ============================================================

for col in feature_columns:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

X_df = X_df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6. FILL MISSING FEATURE VALUES WITH MEDIAN
# ============================================================

print("\nMissing values before imputation:")
print(X_df.isnull().sum())

for col in feature_columns:
    if X_df[col].isnull().any():
        median_val = X_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_df[col] = X_df[col].fillna(median_val)

print("\nMissing values after imputation:")
print(X_df.isnull().sum())


# ============================================================
# 7. CONVERT TO NUMPY
# ============================================================

X = X_df.to_numpy(dtype=float)
y = y_series.to_numpy(dtype=int)

valid_rows = np.isfinite(X).all(axis=1)
X = X[valid_rows]
y = y[valid_rows]

print("\nFinal dataset shape:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 10. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Training completed.")


# ============================================================
# 11. PREDICTIONS
# ============================================================

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 12. METRICS
# ============================================================

accuracy  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test,    y_pred)
f1        = f1_score(y_test,        y_pred)
roc_auc   = roc_auc_score(y_test,   y_pred_proba)

print("\n" + "=" * 75)
print("MODEL EVALUATION METRICS")
print("=" * 75)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Healthy (0)", "Unhealthy (1)"]
))


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
    "Value":  [accuracy, precision, recall, f1, roc_auc]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "metrics.csv"),
    index=False
)

print("\nMetrics saved.")


# ============================================================
# 14. IMAGE 1 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Healthy", "Unhealthy"],
    yticklabels=["Healthy", "Unhealthy"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: confusion_matrix.png")


# ============================================================
# 15. IMAGE 2 - ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Binary)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: roc_curve.png")


# ============================================================
# 16. IMAGE 3 - FEATURE COEFFICIENTS
# ============================================================

plt.figure(figsize=(10, 6))

colors = [
    "green" if c >= 0 else "red"
    for c in model.coef_[0]
]

plt.barh(feature_columns, model.coef_[0], color=colors)
plt.axvline(x=0, linestyle="--", color="black")
plt.xlabel("Coefficient Value")
plt.title("Feature Coefficients - Logistic Regression")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "feature_coefficients.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: feature_coefficients.png")


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
