"""
src/model_report.py  —  AERO-PURE
Trains all models, computes regression + derived/direct classification
metrics, confusion matrices and saves everything to
output/model_report/models.json  +  PNG heatmaps.

Run:  python src/model_report.py
Results are cached; app.py reads the JSON, never retrains on page load.
"""

from __future__ import annotations
import json, time, warnings
import pathlib, sys
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestRegressor
from sklearn.linear_model    import (LinearRegression, Ridge, Lasso,
                                     ElasticNet, LogisticRegression)
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.metrics         import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

# ── Paths (all relative to project root) ──────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV    = ROOT / "dataset" / "PRSA_Data_Aotizhongxin_raw.csv"
OUT_DIR    = ROOT / "output" / "model_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH  = OUT_DIR / "models.json"

FEATURES = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]
TARGET   = "PM2.5"
RS       = 42          # random_state — matches train_model.py


# ── AQI helpers (exact thresholds from Multinomial script) ────

def assign_aqi_binary(pm25: float) -> int:
    """0 = Healthy (PM2.5 < 75),  1 = Unhealthy (PM2.5 >= 75)"""
    return 1 if pm25 >= 75 else 0

def assign_aqi_class(pm25: float) -> int:
    """0=Good <35 | 1=Moderate 35-75 | 2=Unhealthy 75-150 | 3=Hazardous 150+"""
    if pm25 < 35:   return 0
    if pm25 < 75:   return 1
    if pm25 < 150:  return 2
    return 3

BINARY_LABELS    = ["Healthy (PM2.5<75)", "Unhealthy (PM2.5≥75)"]
MULTICLASS_LABELS = ["Good (<35)", "Moderate (35-75)",
                     "Unhealthy (75-150)", "Hazardous (≥150)"]


# ── Data loading & splitting ───────────────────────────────────

def load_data():
    print("Loading dataset …")
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=[TARGET]).copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    X    = df[FEATURES].values
    y_reg = df[TARGET].values
    y_bin = np.array([assign_aqi_binary(v) for v in y_reg])
    y_mc  = np.array([assign_aqi_class(v)  for v in y_reg])

    (X_tr, X_te,
     yr_tr, yr_te,
     yb_tr, yb_te,
     ym_tr, ym_te) = train_test_split(
        X, y_reg, y_bin, y_mc, test_size=0.2, random_state=RS
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    print(f"  Train {X_tr.shape[0]:,}  Test {X_te.shape[0]:,}  "
          f"Features {len(FEATURES)}")
    return (X_tr_sc, X_te_sc,
            yr_tr, yr_te,
            yb_tr, yb_te,
            ym_tr, ym_te,
            len(yr_te))


# ── Confusion-matrix helpers ───────────────────────────────────

def cm_dict(y_true, y_pred, labels) -> dict:
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(labels)))).tolist()
    return {"labels": labels, "matrix": cm}


def cm_png(y_true, y_pred, labels, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(4, len(labels)*1.8),
                                    max(3, len(labels)*1.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(title,        fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def clf_metrics(y_true, y_pred, labels, avg="weighted") -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        labels=list(range(len(labels))),
        output_dict=True, zero_division=0
    )
    return {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":       round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_true,    y_pred, average=avg, zero_division=0)), 4),
        "f1":              round(float(f1_score(y_true,        y_pred, average=avg, zero_division=0)), 4),
        "per_class_report": {
            lbl: {k: round(v, 4) for k, v in report[lbl].items()}
            for lbl in labels if lbl in report
        },
    }


def roc_png(y_true, y_score, model_id: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Binary (Healthy / Unhealthy)")
    ax.legend()
    plt.tight_layout()
    fname = f"{model_id}_roc.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── Build one model entry ──────────────────────────────────────

def build_entry(
    model_id: str, name: str, model_type: str,
    description: str, hyperparams: dict,
    sklearn_model,
    X_tr, X_te,
    yr_tr, yr_te,
    yb_tr, yb_te,
    ym_tr, ym_te,
    test_n: int,
    is_classifier: bool = False,
    binary_target_tr=None,
    binary_target_te=None,
) -> dict:

    t0 = time.perf_counter()

    entry: dict = {
        "id":              model_id,
        "name":            name,
        "type":            model_type,
        "description":     description,
        "hyperparameters": hyperparams,
        "train_time_seconds": 0.0,
    }

    if not is_classifier:
        # ── Regressor ────────────────────────────────────────
        sklearn_model.fit(X_tr, yr_tr)
        y_pred_reg = sklearn_model.predict(X_te)
        t1 = time.perf_counter()

        entry["regression"] = {
            "r2":   round(float(r2_score(yr_te, y_pred_reg)), 4),
            "mae":  round(float(mean_absolute_error(yr_te, y_pred_reg)), 4),
            "mse":  round(float(mean_squared_error(yr_te, y_pred_reg)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yr_te, y_pred_reg))), 4),
        }

        # Derive AQI classes from predicted PM2.5
        yb_pred = np.array([assign_aqi_binary(v) for v in y_pred_reg])
        ym_pred = np.array([assign_aqi_class(v)  for v in y_pred_reg])
        note = ("derived — predicted PM2.5 values converted to AQI classes "
                "using the same thresholds as direct classifiers")

        # Binary derived
        bm = clf_metrics(yb_te, yb_pred, BINARY_LABELS, avg="binary")
        bm["note"] = note
        bm.update(cm_dict(yb_te, yb_pred, BINARY_LABELS))
        bm["confusion_matrix_png"] = cm_png(
            yb_te, yb_pred, BINARY_LABELS,
            f"{name} — Binary Confusion Matrix (derived)",
            f"{model_id}_binary_cm.png")
        entry["binary"] = bm

        # 4-class derived
        mc = clf_metrics(ym_te, ym_pred, MULTICLASS_LABELS)
        mc["note"] = note
        mc.update(cm_dict(ym_te, ym_pred, MULTICLASS_LABELS))
        mc["confusion_matrix_png"] = cm_png(
            ym_te, ym_pred, MULTICLASS_LABELS,
            f"{name} — 4-class Confusion Matrix (derived)",
            f"{model_id}_multiclass_cm.png")
        entry["multiclass"] = mc

    else:
        # ── Classifier (Logistic models) ─────────────────────
        # binary_target_tr/te are either binary (0/1) or multiclass (0-3)
        sklearn_model.fit(X_tr, binary_target_tr)
        y_pred_clf = sklearn_model.predict(X_te)
        t1 = time.perf_counter()

        if model_id == "logistic_binary":
            bm = clf_metrics(binary_target_te, y_pred_clf, BINARY_LABELS, avg="binary")
            bm.update(cm_dict(binary_target_te, y_pred_clf, BINARY_LABELS))
            bm["confusion_matrix_png"] = cm_png(
                binary_target_te, y_pred_clf, BINARY_LABELS,
                f"{name} — Binary Confusion Matrix",
                f"{model_id}_binary_cm.png")
            entry["binary"] = bm

            # ROC-AUC
            if hasattr(sklearn_model, "predict_proba"):
                y_score = sklearn_model.predict_proba(X_te)[:, 1]
            else:
                y_score = sklearn_model.decision_function(X_te)
            entry["roc_auc"] = round(float(roc_auc_score(binary_target_te, y_score)), 4)
            entry["roc_png"] = roc_png(binary_target_te, y_score, model_id)

            # Derive multiclass from probability threshold via pm25 is not available
            # for pure classifier — mark as N/A
            entry["multiclass"] = {"note": "Not applicable for binary-only logistic model."}

        else:  # logistic_multiclass
            mc = clf_metrics(binary_target_te, y_pred_clf, MULTICLASS_LABELS)
            mc.update(cm_dict(binary_target_te, y_pred_clf, MULTICLASS_LABELS))
            mc["confusion_matrix_png"] = cm_png(
                binary_target_te, y_pred_clf, MULTICLASS_LABELS,
                f"{name} — 4-class Confusion Matrix",
                f"{model_id}_multiclass_cm.png")
            entry["multiclass"] = mc
            entry["binary"] = {"note": "Not applicable for multi-class logistic model."}

    entry["train_time_seconds"] = round(t1 - t0, 3)
    print(f"  ✓  {name:<40} ({entry['train_time_seconds']}s)")
    return entry


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  AERO-PURE  model_report.py")
    print("=" * 64)

    (X_tr, X_te,
     yr_tr, yr_te,
     yb_tr, yb_te,
     ym_tr, ym_te,
     test_n) = load_data()

    models_list = []

    # ── Regressors ────────────────────────────────────────────
    reg_specs = [
        ("linear_regression", "Linear Regression",
         "Ordinary Least Squares — finds the best linear fit between features and PM2.5.",
         {"fit_intercept": True},
         LinearRegression()),

        ("ridge", "Ridge Regression",
         "OLS with L2 penalty (alpha=1.0) — shrinks large coefficients to reduce overfitting.",
         {"alpha": 1.0},
         Ridge(alpha=1.0, random_state=RS)),

        ("lasso", "Lasso Regression",
         "OLS with L1 penalty (alpha=0.1) — zeroes out unimportant feature coefficients.",
         {"alpha": 0.1, "max_iter": 10000},
         Lasso(alpha=0.1, max_iter=10000, random_state=RS)),

        ("elasticnet", "ElasticNet",
         "Combined L1+L2 regularisation (alpha=0.1, l1_ratio=0.5).",
         {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000},
         ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RS)),

        ("decision_tree_regressor", "Decision Tree Regressor",
         "Recursive binary splits on feature thresholds (max_depth=5).",
         {"criterion": "squared_error", "max_depth": 5},
         DecisionTreeRegressor(max_depth=5, random_state=RS)),

        ("random_forest", "Random Forest Regressor",
         "Ensemble of 100 decision trees with bootstrap sampling (max_depth=10).",
         {"n_estimators": 100, "max_depth": 10, "bootstrap": True},
         RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=RS, n_jobs=-1)),
    ]

    for (mid, name, desc, hp, model) in reg_specs:
        entry = build_entry(
            mid, name, "regression", desc, hp, model,
            X_tr, X_te, yr_tr, yr_te, yb_tr, yb_te, ym_tr, ym_te, test_n,
            is_classifier=False)
        models_list.append(entry)

    # ── Classifiers ───────────────────────────────────────────
    clf_specs = [
        ("logistic_binary", "Logistic Regression (Binary)",
         "Sigmoid function with threshold 0.5; classifies PM2.5≥75 as Unhealthy.",
         {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
         LogisticRegression(max_iter=1000, random_state=RS),
         yb_tr, yb_te),

        ("logistic_multiclass", "Logistic Regression (Multi-class)",
         "Softmax over 4 AQI classes: Good/Moderate/Unhealthy/Hazardous.",
         {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
         LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RS),
         ym_tr, ym_te),
    ]

    for (mid, name, desc, hp, model, y_tr_c, y_te_c) in clf_specs:
        entry = build_entry(
            mid, name, "classification", desc, hp, model,
            X_tr, X_te, yr_tr, yr_te, yb_tr, yb_te, ym_tr, ym_te, test_n,
            is_classifier=True,
            binary_target_tr=y_tr_c,
            binary_target_te=y_te_c)
        models_list.append(entry)

    # ── Save JSON ─────────────────────────────────────────────
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(models_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved {len(models_list)} model entries → {JSON_PATH}")
    print(f"   PNG heatmaps → {OUT_DIR}")

    # ── Quick summary ─────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Model':<42} {'R²':>7} {'BinAcc':>8} {'4clAcc':>8}")
    print("-" * 70)
    for e in models_list:
        r2     = e.get("regression", {}).get("r2",  "—")
        binacc = e.get("binary",     {}).get("accuracy", "—")
        mcacc  = e.get("multiclass", {}).get("accuracy", "—")
        print(f"{e['name']:<42} {str(r2):>7} {str(binacc):>8} {str(mcacc):>8}")

    print("\nRun  python app.py  to start the dashboard.")


if __name__ == "__main__":
    main()
