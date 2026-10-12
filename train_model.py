"""
train_model.py — AERO-PURE
Trains a Random Forest Regressor on the Aotizhongxin PM2.5 dataset,
evaluates it, and saves the model + scaler to models/.
Run once:  python train_model.py
"""

import os
import json
import pickle
import warnings
import numpy  as np
import pandas as pd

from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree             import DecisionTreeRegressor
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (mean_squared_error, mean_absolute_error,
                                       r2_score, accuracy_score, precision_score,
                                       recall_score, f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_CSV = os.path.join(DATASET_DIR, "PRSA_Data_Aotizhongxin_raw.csv")

FEATURES = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]
TARGET   = "PM2.5"

# ──────────────────────────────────────────────────────────────
# 1. Load and clean
# ──────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(RAW_CSV)
df = df.dropna(subset=[TARGET]).copy()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

X    = df[FEATURES].values
y_reg = df[TARGET].values

print(f"  Rows used: {len(X):,}  |  Features: {len(FEATURES)}")

# AQI class labels
def aqi_class(v):
    if v < 35:   return 0   # Good
    if v < 75:   return 1   # Moderate
    if v < 150:  return 2   # Unhealthy
    return 3                # Hazardous

y_clf = np.array([aqi_class(v) for v in y_reg])
y_bin = (y_reg >= 75).astype(int)

# ──────────────────────────────────────────────────────────────
# 2. Split  (same random_state everywhere for reproducibility)
# ──────────────────────────────────────────────────────────────
RS = 42
Xtr, Xte, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=RS)
_,   _,   yc_tr, yc_te = train_test_split(X, y_clf, test_size=0.2, random_state=RS)
_,   _,   yb_tr, yb_te = train_test_split(X, y_bin, test_size=0.2, random_state=RS)

scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(Xtr)
Xte_sc  = scaler.transform(Xte)

# ──────────────────────────────────────────────────────────────
# 3. Train all regression models
# ──────────────────────────────────────────────────────────────
results = {}

def reg_eval(name, label, model):
    model.fit(Xtr_sc, yr_tr)
    p = model.predict(Xte_sc)
    results[name] = {
        "label": label,
        "type":  "regression",
        "r2":    round(float(r2_score(yr_te, p)), 4),
        "mae":   round(float(mean_absolute_error(yr_te, p)), 2),
        "mse":   round(float(mean_squared_error(yr_te, p)), 2),
        "rmse":  round(float(np.sqrt(mean_squared_error(yr_te, p))), 2),
    }
    print(f"  {label:<35} R²={results[name]['r2']}  MAE={results[name]['mae']}  RMSE={results[name]['rmse']}")

def clf_eval(name, label, model, y_tr, y_te, classes, avg="weighted"):
    model.fit(Xtr_sc, y_tr)
    p  = model.predict(Xte_sc)
    cm = confusion_matrix(y_te, p).tolist()
    fi = {}
    if hasattr(model, "feature_importances_"):
        fi = {FEATURES[i]: round(float(model.feature_importances_[i]), 4)
              for i in range(len(FEATURES))}
    results[name] = {
        "label":              label,
        "type":               "classification",
        "classes":            classes,
        "accuracy":           round(float(accuracy_score(y_te, p)), 4),
        "precision":          round(float(precision_score(y_te, p, average=avg, zero_division=0)), 4),
        "recall":             round(float(recall_score(y_te,    p, average=avg, zero_division=0)), 4),
        "f1":                 round(float(f1_score(y_te,        p, average=avg, zero_division=0)), 4),
        "confusion_matrix":   cm,
        "feature_importance": fi,
    }
    print(f"  {label:<35} Acc={results[name]['accuracy']}  F1={results[name]['f1']}")

print("\n── Regression ──────────────────────────────────────────")
reg_eval("linear_regression",      "Linear Regression",       LinearRegression())
reg_eval("ridge",                  "Ridge Regression",        Ridge(alpha=1.0))
reg_eval("lasso",                  "Lasso Regression",        Lasso(alpha=0.1, max_iter=10000))
reg_eval("elasticnet",             "ElasticNet",              ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
reg_eval("decision_tree_regressor","Decision Tree Regressor", DecisionTreeRegressor(max_depth=5, random_state=RS))
reg_eval("random_forest",         "Random Forest Regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RS, n_jobs=-1))

print("\n── Classification ───────────────────────────────────────")
clf_eval("logistic_binary",        "Logistic Regression (Binary)",
         LogisticRegression(max_iter=1000, random_state=RS),
         yb_tr, yb_te, ["Healthy (PM2.5<75)","Unhealthy (PM2.5≥75)"], "binary")

clf_eval("logistic_multiclass",    "Logistic Regression (Multi-class)",
         LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RS),
         yc_tr, yc_te, ["Good","Moderate","Unhealthy","Hazardous"])

clf_eval("decision_tree_classifier","Decision Tree Classifier",
         DecisionTreeClassifier(max_depth=5, random_state=RS),
         yc_tr, yc_te, ["Good","Moderate","Unhealthy","Hazardous"])

# ──────────────────────────────────────────────────────────────
# 4. Save model_results.json
# ──────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "model_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nModel results saved → {json_path}")

# ──────────────────────────────────────────────────────────────
# 5. Save best regression model (Random Forest) + scaler
# ──────────────────────────────────────────────────────────────
# Re-train on full training set for deployment
best_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=RS, n_jobs=-1)
best_model.fit(Xtr_sc, yr_tr)

with open(os.path.join(MODEL_DIR, "pm25_model.pkl"),  "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, "pm25_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save feature list so app.py can verify alignment
with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
    json.dump(FEATURES, f)

print(f"Trained model saved  → {os.path.join(MODEL_DIR, 'pm25_model.pkl')}")
print(f"Scaler saved         → {os.path.join(MODEL_DIR, 'pm25_scaler.pkl')}")

print("\n" + "="*55)
print("Training complete.")
print("="*55)
