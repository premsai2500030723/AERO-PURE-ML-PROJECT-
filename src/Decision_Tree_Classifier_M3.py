import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "Decision_Tree_Classifier_M3"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("DECISION TREE CLASSIFIER")
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
# 3. CREATE AQI CLASS TARGET
# ============================================================

data = data.dropna(subset=["PM2.5"])


def assign_aqi_class(pm25):
    if pm25 < 35:
        return 0    # Good
    elif pm25 < 75:
        return 1    # Moderate
    elif pm25 < 150:
        return 2    # Unhealthy
    else:
        return 3    # Hazardous


data["AQI_Class"] = data["PM2.5"].apply(assign_aqi_class)

class_names = ["Good", "Moderate", "Unhealthy", "Hazardous"]

print("\nAQI Class Distribution:")
print(data["AQI_Class"].value_counts().sort_index())

print("\nClass Labels:")
print("0 = Good       (PM2.5 < 35)")
print("1 = Moderate   (35 <= PM2.5 < 75)")
print("2 = Unhealthy  (75 <= PM2.5 < 150)")
print("3 = Hazardous  (PM2.5 >= 150)")


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

target_column = "AQI_Class"

X_df     = data[feature_columns].copy()
y_series = data[target_column].copy()


# ============================================================
# 5. CONVERT TO NUMERIC
# ============================================================

for col in feature_columns:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

X_df = X_df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6. FILL MISSING VALUES WITH MEDIAN
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
# 10. TRAIN DECISION TREE
# ============================================================

print("\nTraining Decision Tree Classifier...")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Training completed.")
print("Tree depth     :", model.get_depth())
print("Number of leaves:", model.get_n_leaves())


# ============================================================
# 11. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 12. METRICS
# ============================================================

accuracy  = accuracy_score(y_test,  y_pred)
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
    target_names=class_names,
    zero_division=0
))


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Score_Weighted"],
    "Value":  [accuracy, precision, recall, f1]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "metrics.csv"),
    index=False
)

print("Metrics saved.")


# ============================================================
# 14. IMAGE 1 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

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
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: confusion_matrix.png")


# ============================================================
# 15. IMAGE 2 - DECISION TREE PLOT
# ============================================================

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=feature_columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)

plt.title("Decision Tree - Air Quality Classification (depth=3 shown)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "decision_tree_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: decision_tree_plot.png")


# ============================================================
# 16. IMAGE 3 - FEATURE IMPORTANCE
# ============================================================

importances = model.feature_importances_

sorted_idx  = np.argsort(importances)[::-1]

sorted_features    = [feature_columns[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10, 6))

plt.bar(sorted_features, sorted_importances, color="steelblue")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance - Decision Tree Classifier")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: feature_importance.png")


# ============================================================
# 17. IMAGE 4 - CLASS DISTRIBUTION
# ============================================================

class_counts = pd.Series(y).value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    class_names,
    class_counts.values,
    color=["green", "yellow", "orange", "red"]
)

plt.xlabel("AQI Class")
plt.ylabel("Count")
plt.title("AQI Class Distribution - Air Quality Dataset")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved: class_distribution.png")


# ============================================================
# 18. SAVE TREE TEXT
# ============================================================

tree_text = export_text(
    model,
    feature_names=feature_columns
)

with open(os.path.join(OUTPUT_FOLDER, "decision_tree_rules.txt"), "w") as f:
    f.write(tree_text)

print("Tree rules saved: decision_tree_rules.txt")


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED TO:")
print(OUTPUT_FOLDER)
print("=" * 75)

print("\n1. confusion_matrix.png")
print("2. decision_tree_plot.png")
print("3. feature_importance.png")
print("4. class_distribution.png")
print("5. metrics.csv")
print("6. decision_tree_rules.txt")

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
