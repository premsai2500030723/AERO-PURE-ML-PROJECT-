"""
AdaBoost_Classifier_M3.py
==========================
AdaBoost Classifier on the Air Quality Dataset.
Creates 4 AQI categories from PM2.5 values:
    0 = Good       (PM2.5 < 35)
    1 = Moderate   (35 <= PM2.5 < 75)
    2 = Unhealthy  (75 <= PM2.5 < 150)
    3 = Hazardous  (PM2.5 >= 150)

Output
------
output/AdaBoost_Classifier_M3/

Usage
-----
    python src/AdaBoost_Classifier_M3.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = BASE_DIR / "output" / "AdaBoost_Classifier_M3"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("ADABOOST CLASSIFIER")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])


# ============================================================
# 3. CREATE AQI CLASS TARGET
# ============================================================

data = data.dropna(subset=["PM2.5"]).copy()


def assign_aqi_class(pm25):
    if pm25 < 35:   return 0
    elif pm25 < 75:  return 1
    elif pm25 < 150: return 2
    else:            return 3


data["AQI_Class"] = data["PM2.5"].apply(assign_aqi_class)

CLASS_NAMES = ["Good", "Moderate", "Unhealthy", "Hazardous"]

print("\nAQI Class Distribution:")
print(data["AQI_Class"].value_counts().sort_index())


# ============================================================
# 4. FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

X_df     = data[FEATURE_COLUMNS].copy()
y_series = data["AQI_Class"].copy()


# ============================================================
# 5. CONVERT TO NUMERIC AND FILL MISSING
# ============================================================

for col in FEATURE_COLUMNS:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df[col] = X_df[col].fillna(X_df[col].median())

X_df = X_df.replace([np.inf, -np.inf], np.nan)

for col in FEATURE_COLUMNS:
    X_df[col] = X_df[col].fillna(X_df[col].median())


# ============================================================
# 6. CONVERT TO NUMPY
# ============================================================

X = X_df.to_numpy(dtype=float)
y = y_series.to_numpy(dtype=int)

valid = np.isfinite(X).all(axis=1)
X = X[valid]
y = y[valid]

print("\nFinal dataset shape:")
print("X:", X.shape, "  y:", y.shape)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("X_train:", X_train.shape, "  X_test:", X_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 9. TRAIN ADABOOST CLASSIFIER
# ============================================================

print("\nTraining AdaBoost Classifier …")

base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Training completed.")
print("Number of estimators:", model.n_estimators)


# ============================================================
# 10. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 11. METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test,    y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test,        y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 75)
print("MODEL EVALUATION METRICS")
print("=" * 75)
print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision (wtd)   : {precision:.4f}")
print(f"Recall    (wtd)   : {recall:.4f}")
print(f"F1 Score  (wtd)   : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Score_Weighted"],
    "Value":  [accuracy, precision, recall, f1]
})

metrics_df.to_csv(OUTPUT_FOLDER / "metrics.csv", index=False)
print("Metrics saved.")


# ============================================================
# 13. IMAGE 1 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — AdaBoost Classifier")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: confusion_matrix.png")


# ============================================================
# 14. IMAGE 2 - FEATURE IMPORTANCES
# ============================================================

importances    = model.feature_importances_
sorted_idx     = np.argsort(importances)[::-1]
sorted_feats   = [FEATURE_COLUMNS[i] for i in sorted_idx]
sorted_imps    = importances[sorted_idx]

plt.figure(figsize=(10, 6))
plt.bar(sorted_feats, sorted_imps, color="steelblue")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance — AdaBoost Classifier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: feature_importance.png")


# ============================================================
# 15. IMAGE 3 - CLASS DISTRIBUTION
# ============================================================

class_counts = pd.Series(y).value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(CLASS_NAMES, class_counts.values,
        color=["green", "yellow", "orange", "red"])
plt.xlabel("AQI Class")
plt.ylabel("Count")
plt.title("AQI Class Distribution - Air Quality Dataset")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: class_distribution.png")


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(str(OUTPUT_FOLDER))
print("=" * 75)
print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
