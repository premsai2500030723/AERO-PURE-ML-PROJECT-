"""
XGBoost_Clasifier_M3.py
========================
XGBoost Classifier on the Air Quality Dataset.
Classifies PM2.5 into 4 AQI categories:
    0 = Good, 1 = Moderate, 2 = Unhealthy, 3 = Hazardous

Output
------
output/XGBoost_Classifier_M3/

Usage
-----
    python src/XGBoost_Clasifier_M3.py

Note
----
Requires: pip install xgboost
Falls back to GradientBoostingClassifier if xgboost is not installed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Try XGBoost, fall back gracefully
try:
    from xgboost import XGBClassifier
    USING_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USING_XGB = False
    print("WARNING: xgboost not installed. Using GradientBoostingClassifier as fallback.")
    print("Install with: pip install xgboost")


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR      = pathlib.Path(__file__).resolve().parent.parent
INPUT_FILE    = BASE_DIR / "dataset" / "PRSA_Data_Aotizhongxin_raw.csv"
OUTPUT_FOLDER = BASE_DIR / "output" / "XGBoost_Classifier_M3"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

MODEL_NAME = "XGBoost Classifier" if USING_XGB else "GradientBoosting Classifier (XGBoost fallback)"

print("=" * 75)
print(MODEL_NAME.upper())
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

data = pd.read_csv(INPUT_FILE)
data = data.dropna(subset=["PM2.5"]).copy()
print(f"\nRows: {data.shape[0]}   Columns: {data.shape[1]}")


# ============================================================
# 3. AQI CLASS TARGET
# ============================================================

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
# 4. FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

X_df = data[FEATURE_COLUMNS].copy()
y    = data["AQI_Class"].to_numpy(dtype=int)

for col in FEATURE_COLUMNS:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df[col] = X_df[col].fillna(X_df[col].median())

X = X_df.to_numpy(dtype=float)
valid = np.isfinite(X).all(axis=1)
X, y  = X[valid], y[valid]

print(f"\nFinal shape — X: {X.shape}  y: {y.shape}")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print(f"\nTraining {MODEL_NAME} …")

if USING_XGB:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
else:
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

model.fit(X_train_sc, y_train)
print("Training completed.")


# ============================================================
# 7. EVALUATE
# ============================================================

y_pred = model.predict(X_test_sc)

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
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))


# ============================================================
# 8. SAVE METRICS
# ============================================================

pd.DataFrame({
    "Metric": ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Score_Weighted"],
    "Value":  [accuracy, precision, recall, f1]
}).to_csv(OUTPUT_FOLDER / "metrics.csv", index=False)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {MODEL_NAME}")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: confusion_matrix.png")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importances = model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar([FEATURE_COLUMNS[i] for i in sorted_idx],
        importances[sorted_idx], color="steelblue")
plt.xticks(rotation=45)
plt.xlabel("Feature"); plt.ylabel("Importance")
plt.title(f"Feature Importance — {MODEL_NAME}")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: feature_importance.png")


# ============================================================
# 11. CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(CLASS_NAMES, pd.Series(y).value_counts().sort_index().values,
        color=["green", "yellow", "orange", "red"])
plt.xlabel("AQI Class"); plt.ylabel("Count")
plt.title("AQI Class Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Image saved: class_distribution.png")


print("\n" + "=" * 75)
print(f"OUTPUTS SAVED TO: {OUTPUT_FOLDER}")
print("=" * 75)
print(f"\nModel used: {MODEL_NAME}")
print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
