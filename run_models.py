"""Run all models and save real metrics to output/model_results.json"""

import pandas as pd
import numpy as np
import os
import json
import warnings

from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree          import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("dataset/PRSA_Data_Aotizhongxin_raw.csv")
df = df.dropna(subset=["PM2.5"]).copy()

feature_cols = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=feature_cols)

X    = df[feature_cols].values
y_reg = df["PM2.5"].values

def aqi(v):
    if v < 35:  return 0
    elif v < 75:  return 1
    elif v < 150: return 2
    else:         return 3

y_clf = np.array([aqi(v) for v in y_reg])
y_bin = (y_reg >= 75).astype(int)

X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_,    _,    yc_tr, yc_te = train_test_split(X, y_clf, test_size=0.2, random_state=42)
_,    _,    yb_tr, yb_te = train_test_split(X, y_bin, test_size=0.2, random_state=42)

sc    = StandardScaler()
Xtr   = sc.fit_transform(X_tr)
Xte   = sc.transform(X_te)

results = {}

# ── Regression models ─────────────────────────────────────────
def reg_metrics(name, model, label):
    model.fit(Xtr, yr_tr)
    p = model.predict(Xte)
    results[name] = {
        "label": label,
        "type":  "regression",
        "r2":    round(r2_score(yr_te, p), 4),
        "mae":   round(mean_absolute_error(yr_te, p), 2),
        "mse":   round(mean_squared_error(yr_te, p), 2),
        "rmse":  round(float(np.sqrt(mean_squared_error(yr_te, p))), 2),
    }

reg_metrics("linear_regression",      LinearRegression(),                          "Linear Regression")
reg_metrics("ridge",                   Ridge(alpha=1.0),                            "Ridge Regression")
reg_metrics("lasso",                   Lasso(alpha=0.1, max_iter=10000),            "Lasso Regression")
reg_metrics("elasticnet",              ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), "ElasticNet")
reg_metrics("decision_tree_regressor", DecisionTreeRegressor(max_depth=5, random_state=42), "Decision Tree Regressor")

# ── Classification models ──────────────────────────────────────
def clf_metrics(name, model, label, y_tr, y_te, classes, average="weighted"):
    model.fit(Xtr, y_tr)
    p  = model.predict(Xte)
    cm = confusion_matrix(y_te, p).tolist()
    fi = {}
    if hasattr(model, "feature_importances_"):
        fi = {feature_cols[i]: round(float(model.feature_importances_[i]), 4)
              for i in range(len(feature_cols))}
    results[name] = {
        "label":            label,
        "type":             "classification",
        "classes":          classes,
        "accuracy":         round(accuracy_score(y_te, p), 4),
        "precision":        round(precision_score(y_te, p, average=average, zero_division=0), 4),
        "recall":           round(recall_score(y_te,    p, average=average, zero_division=0), 4),
        "f1":               round(f1_score(y_te,        p, average=average, zero_division=0), 4),
        "confusion_matrix": cm,
        "feature_importance": fi,
    }

clf_metrics("logistic_binary",       LogisticRegression(max_iter=1000, random_state=42),
            "Logistic Regression (Binary)", yb_tr, yb_te,
            ["Healthy (PM2.5<75)", "Unhealthy (PM2.5>=75)"], average="binary")

clf_metrics("logistic_multiclass",   LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
            "Logistic Regression (Multi-class)", yc_tr, yc_te,
            ["Good", "Moderate", "Unhealthy", "Hazardous"])

clf_metrics("decision_tree_classifier", DecisionTreeClassifier(max_depth=5, random_state=42),
            "Decision Tree Classifier", yc_tr, yc_te,
            ["Good", "Moderate", "Unhealthy", "Hazardous"])

# ── Save ──────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
with open("output/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("MODEL RESULTS SAVED → output/model_results.json")
print("=" * 60)
for k, v in results.items():
    if v["type"] == "regression":
        print(f"{v['label']:<35} R2={v['r2']}  MAE={v['mae']}  RMSE={v['rmse']}")
    else:
        print(f"{v['label']:<35} Acc={v['accuracy']}  F1={v['f1']}")
