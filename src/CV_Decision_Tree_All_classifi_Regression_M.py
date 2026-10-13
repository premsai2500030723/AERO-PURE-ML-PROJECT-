"""
CV_Decision_Tree_All_classifi_Regression_M.py
===============================================
Cross-Validation on Decision Tree for both Classification and Regression
on the Air Quality Dataset.

Classification : Predicts AQI Class (Good/Moderate/Unhealthy/Hazardous)
Regression     : Predicts exact PM2.5 value

Uses K-Fold Cross Validation (k=5) for both tasks.

Output
------
output/CV_Decision_Tree_All_Classifi_Regression_M/

Usage
-----
    python src/CV_Decision_Tree_All_classifi_Regression_M.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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
    "CV_Decision_Tree_All_Classifi_Regression_M"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("CROSS-VALIDATION - DECISION TREE (CLASSIFICATION + REGRESSION)")
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
# 3. FEATURE COLUMNS
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

print("\nFeatures:", feature_columns)


# ============================================================
# 4. PREPARE CLASSIFICATION TARGET
# ============================================================

data_clf = data.dropna(subset=["PM2.5"]).copy()


def assign_aqi_class(pm25):
    if pm25 < 35:
        return 0
    elif pm25 < 75:
        return 1
    elif pm25 < 150:
        return 2
    else:
        return 3


data_clf["AQI_Class"] = data_clf["PM2.5"].apply(assign_aqi_class)

class_names = ["Good", "Moderate", "Unhealthy", "Hazardous"]

print("\nAQI Class Distribution:")
print(data_clf["AQI_Class"].value_counts().sort_index())


# ============================================================
# 5. PREPARE REGRESSION TARGET
# ============================================================

data_reg = data.dropna(subset=["PM2.5"]).copy()


# ============================================================
# 6. HELPER — CLEAN FEATURES
# ============================================================

def prepare_features(source_df, features):

    X_df = source_df[features].copy()

    for col in features:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    for col in features:
        if X_df[col].isnull().any():
            med = X_df[col].median()
            X_df[col] = X_df[col].fillna(med if not pd.isna(med) else 0.0)

    return X_df.to_numpy(dtype=float)


# ============================================================
# 7. CLASSIFICATION — PREPARE X AND y
# ============================================================

X_clf = prepare_features(data_clf, feature_columns)
y_clf = data_clf["AQI_Class"].to_numpy(dtype=int)

valid_clf = np.isfinite(X_clf).all(axis=1)
X_clf = X_clf[valid_clf]
y_clf = y_clf[valid_clf]

print("\nClassification dataset shape:")
print("X:", X_clf.shape)
print("y:", y_clf.shape)


# ============================================================
# 8. REGRESSION — PREPARE X AND y
# ============================================================

X_reg = prepare_features(data_reg, feature_columns)
y_reg = data_reg["PM2.5"].to_numpy(dtype=float)

valid_reg = np.isfinite(X_reg).all(axis=1) & np.isfinite(y_reg)
X_reg = X_reg[valid_reg]
y_reg = y_reg[valid_reg]

print("\nRegression dataset shape:")
print("X:", X_reg.shape)
print("y:", y_reg.shape)


# ============================================================
# 9. SCALE FEATURES
# ============================================================

scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)


# ============================================================
# 10. K-FOLD CROSS VALIDATION SETUP
# ============================================================

K_FOLDS = 5
kfold   = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

print("\n" + "=" * 75)
print(f"CROSS-VALIDATION — K = {K_FOLDS} FOLDS")
print("=" * 75)


# ============================================================
# 11. CLASSIFICATION — CROSS VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("PART 1 — DECISION TREE CLASSIFICATION (CV)")
print("=" * 75)

clf_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

clf_cv_results = cross_validate(
    clf_model,
    X_clf_scaled,
    y_clf,
    cv=kfold,
    scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
    return_train_score=True
)

print("\nClassification Cross-Validation Results:")
print("-" * 50)

for fold in range(K_FOLDS):
    print(
        f"Fold {fold+1} | "
        f"Train Acc: {clf_cv_results['train_accuracy'][fold]:.4f} | "
        f"Test Acc : {clf_cv_results['test_accuracy'][fold]:.4f} | "
        f"F1: {clf_cv_results['test_f1_weighted'][fold]:.4f}"
    )

print("\nMean Scores (Classification):")
print(f"Accuracy  : {clf_cv_results['test_accuracy'].mean():.4f} ± {clf_cv_results['test_accuracy'].std():.4f}")
print(f"F1 Score  : {clf_cv_results['test_f1_weighted'].mean():.4f} ± {clf_cv_results['test_f1_weighted'].std():.4f}")
print(f"Precision : {clf_cv_results['test_precision_weighted'].mean():.4f}")
print(f"Recall    : {clf_cv_results['test_recall_weighted'].mean():.4f}")


# ============================================================
# 12. REGRESSION — CROSS VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("PART 2 — DECISION TREE REGRESSION (CV)")
print("=" * 75)

reg_model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    random_state=42
)

reg_cv_results = cross_validate(
    reg_model,
    X_reg_scaled,
    y_reg,
    cv=kfold,
    scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=True
)

print("\nRegression Cross-Validation Results:")
print("-" * 50)

for fold in range(K_FOLDS):
    print(
        f"Fold {fold+1} | "
        f"Train R2: {reg_cv_results['train_r2'][fold]:.4f} | "
        f"Test R2 : {reg_cv_results['test_r2'][fold]:.4f} | "
        f"MSE: {abs(reg_cv_results['test_neg_mean_squared_error'][fold]):.4f}"
    )

print("\nMean Scores (Regression):")
print(f"R2 Score  : {reg_cv_results['test_r2'].mean():.4f} ± {reg_cv_results['test_r2'].std():.4f}")
print(f"MSE       : {abs(reg_cv_results['test_neg_mean_squared_error'].mean()):.4f}")
print(f"MAE       : {abs(reg_cv_results['test_neg_mean_absolute_error'].mean()):.4f}")


# ============================================================
# 13. FINAL TRAIN-TEST EVALUATION — CLASSIFICATION
# ============================================================

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf_scaled, y_clf,
    test_size=0.2, random_state=42, stratify=y_clf
)

clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)

print("\n" + "=" * 75)
print("FINAL TEST SET - CLASSIFICATION METRICS")
print("=" * 75)
print(classification_report(y_test_c, y_pred_c, target_names=class_names, zero_division=0))


# ============================================================
# 14. FINAL TRAIN-TEST EVALUATION — REGRESSION
# ============================================================

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg,
    test_size=0.2, random_state=42
)

reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

test_mse  = mean_squared_error(y_test_r, y_pred_r)
test_rmse = np.sqrt(test_mse)
test_mae  = mean_absolute_error(y_test_r, y_pred_r)
test_r2   = r2_score(y_test_r, y_pred_r)

print("\n" + "=" * 75)
print("FINAL TEST SET - REGRESSION METRICS")
print("=" * 75)
print(f"MSE   : {test_mse:.4f}")
print(f"RMSE  : {test_rmse:.4f}")
print(f"MAE   : {test_mae:.4f}")
print(f"R2    : {test_r2:.4f}")


# ============================================================
# 15. SAVE METRICS
# ============================================================

clf_metrics_df = pd.DataFrame({
    "Fold":      list(range(1, K_FOLDS + 1)),
    "Train_Acc": clf_cv_results["train_accuracy"],
    "Test_Acc":  clf_cv_results["test_accuracy"],
    "Test_F1":   clf_cv_results["test_f1_weighted"]
})

clf_metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "classification_cv_metrics.csv"),
    index=False
)

reg_metrics_df = pd.DataFrame({
    "Fold":      list(range(1, K_FOLDS + 1)),
    "Train_R2":  reg_cv_results["train_r2"],
    "Test_R2":   reg_cv_results["test_r2"],
    "Test_MSE":  abs(reg_cv_results["test_neg_mean_squared_error"])
})

reg_metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "regression_cv_metrics.csv"),
    index=False
)

print("\nMetrics saved.")


# ============================================================
# 16. IMAGE 1 - CLASSIFICATION CV ACCURACY PER FOLD
# ============================================================

folds = list(range(1, K_FOLDS + 1))

plt.figure(figsize=(9, 5))

plt.plot(folds, clf_cv_results["train_accuracy"], marker="o", label="Train Accuracy")
plt.plot(folds, clf_cv_results["test_accuracy"],  marker="o", label="Test Accuracy")

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Decision Tree Classification — CV Accuracy per Fold")
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "clf_cv_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: clf_cv_accuracy.png")


# ============================================================
# 17. IMAGE 2 - REGRESSION CV R2 PER FOLD
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(folds, reg_cv_results["train_r2"], marker="o", label="Train R2")
plt.plot(folds, reg_cv_results["test_r2"],  marker="o", label="Test R2")

plt.xlabel("Fold")
plt.ylabel("R2 Score")
plt.title("Decision Tree Regression — CV R2 Score per Fold")
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "reg_cv_r2.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: reg_cv_r2.png")


# ============================================================
# 18. IMAGE 3 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test_c, y_pred_c)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree Classifier (CV)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: confusion_matrix.png")


# ============================================================
# 19. IMAGE 4 - ACTUAL VS PREDICTED REGRESSION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test_r, y_pred_r, alpha=0.3, s=5, color="steelblue")

min_val = min(y_test_r.min(), y_pred_r.min())
max_val = max(y_test_r.max(), y_pred_r.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", label="Perfect Prediction")

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 - Decision Tree Regression (CV)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "actual_vs_predicted_regression.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: actual_vs_predicted_regression.png")


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\n1. clf_cv_accuracy.png")
print("2. reg_cv_r2.png")
print("3. confusion_matrix.png")
print("4. actual_vs_predicted_regression.png")
print("5. classification_cv_metrics.csv")
print("6. regression_cv_metrics.csv")

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
