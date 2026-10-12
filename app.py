"""
app.py  —  AERO-PURE Air Quality ML Dashboard
Flask web application — all paths relative to this file.
Run:  python app.py
"""

import os
import re
import json
import pickle
import logging
import functools
import numpy  as np
import pandas as pd

from flask import (
    Flask, render_template, request,
    send_from_directory, redirect, url_for, abort, jsonify
)

# ─────────────────────────────────────────────────────────────
# Paths  (all relative to this file — works on any machine)
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RAW_CSV      = os.path.join(DATASET_DIR, "PRSA_Data_Aotizhongxin_raw.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

APP_VERSION  = "2.0.0"

# ─────────────────────────────────────────────────────────────
# Flask setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "aq_ml_aero_pure_2026"
app.jinja_env.globals.update(zip=zip, enumerate=enumerate, len=len,
                              abs=abs, round=round, int=int, str=str,
                              min=min, max=max)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Simple in-process cache (avoids reloading 12 MB CSV on
# every request without requiring Redis / memcached)
# ─────────────────────────────────────────────────────────────
_CACHE: dict = {}

def _cached(key, fn, *args, **kwargs):
    if key not in _CACHE:
        _CACHE[key] = fn(*args, **kwargs)
    return _CACHE[key]

# ═════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═════════════════════════════════════════════════════════════

def load_raw() -> pd.DataFrame | None:
    """Load raw dataset once and cache it."""
    try:
        return _cached("raw_df", pd.read_csv, RAW_CSV)
    except Exception as exc:
        log.error("Cannot load raw CSV: %s", exc)
        return None


def list_dataset_files() -> list[dict]:
    files = []
    for fname in sorted(os.listdir(DATASET_DIR)):
        if fname.endswith(".csv"):
            fpath = os.path.join(DATASET_DIR, fname)
            try:
                df    = pd.read_csv(fpath, nrows=1)
                nrows = sum(1 for _ in open(fpath)) - 1
                ncols = len(df.columns)
            except Exception:
                nrows, ncols = 0, 0
            files.append({"name": fname,
                           "rows": nrows,
                           "cols": ncols,
                           "size_kb": round(os.path.getsize(fpath)/1024, 1)})
    return files


def load_dataframe_safe(filename: str) -> pd.DataFrame | None:
    fname = os.path.basename(filename)          # no path traversal
    path  = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.error("Cannot load %s: %s", fname, exc)
        return None


def get_col_stats(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    stats = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        stats.append({
            "name":        col,
            "dtype":       str(df[col].dtype),
            "missing":     missing,
            "missing_pct": round(missing / total * 100, 2) if total else 0,
            "unique":      int(df[col].nunique()),
        })
    return stats


def real_dashboard_stats() -> dict:
    df = load_raw()
    if df is None:
        return {}
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    missing  = int(df.isnull().sum().sum())
    return {
        "total_rows":    f"{len(df):,}",
        "total_cols":    len(df.columns),
        "numeric_cols":  len(num_cols),
        "cat_cols":      len(cat_cols),
        "missing":       f"{missing:,}",
        "missing_pct":   round(missing / (len(df) * len(df.columns)) * 100, 2),
        "duplicates":    int(df.duplicated().sum()),
        "year_start":    int(df["year"].min()),
        "year_end":      int(df["year"].max()),
        "charts":        len(list_output_images()),
        "datasets":      len(list_dataset_files()),
        "pm25_mean":     round(float(df["PM2.5"].mean()), 2),
        "pm25_max":      round(float(df["PM2.5"].max()), 2),
        "pm25_min":      round(float(df["PM2.5"].min()), 2),
        "temp_mean":     round(float(df["TEMP"].mean()),  2),
        "co_mean":       round(float(df["CO"].mean()),    2),
        "no2_mean":      round(float(df["NO2"].mean()),   2),
    }


# ═════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════

# Columns that are NOT meaningful for analysis (row index)
_SKIP_COLS = {"no"}

# Chart categories — order matters for display
_CATEGORIES = [
    "All",
    "Histogram",
    "Boxplot",
    "Outlier",
    "Correlation",
    "Scatter",
    "Pair Plot",
    "Trend",
    "Count Plot",
    "Missing Values",
    "Other",
]

def _chart_label(fname: str) -> str:
    """Human-readable label from filename."""
    name = fname.replace(".png", "")
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip()


def _chart_category(fname: str) -> str:
    low = fname.lower()
    col_prefix = fname.split("_")[0].lower()
    if col_prefix in _SKIP_COLS:
        return None                   # exclude row-number charts
    if "histogram" in low:            return "Histogram"
    if "boxplot"   in low:            return "Boxplot"
    if "outlier"   in low:            return "Outlier"
    if "heatmap"   in low or "correlation" in low:  return "Correlation"
    if "scatter"   in low or "vs_" in low:           return "Scatter"
    if "pairplot"  in low or "pair" in low:           return "Pair Plot"
    if "trend"     in low or "monthly" in low or "yearly" in low: return "Trend"
    if "countplot" in low or "count" in low:          return "Count Plot"
    if "missing"   in low:            return "Missing Values"
    return "Other"


def list_output_images() -> list[dict]:
    """Return all valid chart dicts from output/."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    result = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        cat = _chart_category(fname)
        if cat is None:
            continue            # skip meaningless charts (No_* columns)
        result.append({
            "filename": fname,
            "label":    _chart_label(fname),
            "category": cat,
        })
    return result


def categorise_images(images: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {c: [] for c in _CATEGORIES}
    for img in images:
        groups["All"].append(img)
        cat = img["category"]
        if cat in groups:
            groups[cat].append(img)
        else:
            groups["Other"].append(img)
    return {k: v for k, v in groups.items() if v}


# ═════════════════════════════════════════════════════════════
# REPORT HELPERS
# ═════════════════════════════════════════════════════════════

def load_missing_report() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "Missing_Values_Report.csv")
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        return []


def load_corr_top(col: str = "PM2.5", n: int = 15) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "Correlation_Matrix.csv")
    if not os.path.exists(path):
        return []
    try:
        df   = pd.read_csv(path, index_col=0)
        if col not in df.columns:
            return []
        corr = (df[col].drop(col, errors="ignore")
                       .sort_values(key=abs, ascending=False)
                       .head(n))
        return [{"feature": k, "value": round(float(v), 4)}
                for k, v in corr.items()]
    except Exception:
        return []


def list_csv_reports() -> list[dict]:
    reports = []
    candidates = {
        "Missing_Values_Report.csv":
            "Missing value counts and percentages per column.",
        "Correlation_Matrix.csv":
            "Pairwise Pearson correlation matrix for all numeric columns.",
        "Statistical_Summary.csv":
            "Descriptive statistics: mean, std, min, max, quartiles.",
    }
    for fname, desc in candidates.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            reports.append({
                "name":     fname.replace("_", " ").replace(".csv", ""),
                "filename": fname,
                "desc":     desc,
                "size_kb":  round(os.path.getsize(fpath) / 1024, 1),
            })
    return reports


def load_stat_summary() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "Statistical_Summary.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, index_col=0)
        rows = []
        for stat, row in df.iterrows():
            rows.append({"stat": stat,
                         **{col: (round(float(v), 4) if pd.notna(v) else "—")
                            for col, v in row.items()}})
        return rows
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════
# MODEL HELPERS
# ═════════════════════════════════════════════════════════════

def load_model_results() -> dict:
    path = os.path.join(OUTPUT_DIR, "model_results.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


_TRAINED_MODEL  = None
_TRAINED_SCALER = None
_MODEL_FEATURES = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]

def _load_trained_model():
    global _TRAINED_MODEL, _TRAINED_SCALER
    mpath = os.path.join(MODEL_DIR, "pm25_model.pkl")
    spath = os.path.join(MODEL_DIR, "pm25_scaler.pkl")
    if os.path.exists(mpath) and os.path.exists(spath):
        with open(mpath, "rb") as f: _TRAINED_MODEL  = pickle.load(f)
        with open(spath, "rb") as f: _TRAINED_SCALER = pickle.load(f)
        log.info("Trained model loaded from %s", MODEL_DIR)
    else:
        log.warning("No trained model found at %s — predictions will be unavailable.", MODEL_DIR)

_load_trained_model()


def predict_pm25(form: dict) -> tuple[float | None, str | None]:
    """Return (predicted_pm25, error_message)."""
    if _TRAINED_MODEL is None or _TRAINED_SCALER is None:
        return None, "No trained model available. Run train_model.py first."
    try:
        vals = []
        for feat in _MODEL_FEATURES:
            raw = form.get(feat.lower(), "")
            if raw == "" or raw is None:
                return None, f"Missing required field: {feat}"
            vals.append(float(raw))
        X = np.array(vals).reshape(1, -1)
        X_scaled = _TRAINED_SCALER.transform(X)
        pred = float(_TRAINED_MODEL.predict(X_scaled)[0])
        pred = max(0.0, round(pred, 2))
        return pred, None
    except ValueError as exc:
        return None, f"Invalid input: {exc}"
    except Exception as exc:
        log.error("Prediction error: %s", exc)
        return None, "Prediction failed — check inputs."


def aqi_category(pm25: float) -> dict:
    """
    US EPA PM2.5 breakpoints (24-hr average).
    Returns label, colour, description.
    """
    if pm25 <= 12.0:
        return {"label":"Good",            "color":"#22c55e","bg":"#dcfce7",
                "desc":"Air quality is satisfactory and poses little or no risk."}
    if pm25 <= 35.4:
        return {"label":"Moderate",         "color":"#eab308","bg":"#fef9c3",
                "desc":"Acceptable quality; sensitive individuals may experience minor effects."}
    if pm25 <= 55.4:
        return {"label":"Unhealthy for Sensitive Groups","color":"#f97316","bg":"#ffedd5",
                "desc":"Sensitive groups (elderly, children, asthma sufferers) may be affected."}
    if pm25 <= 150.4:
        return {"label":"Unhealthy",        "color":"#ef4444","bg":"#fee2e2",
                "desc":"Everyone may experience health effects; sensitive groups more severely."}
    if pm25 <= 250.4:
        return {"label":"Very Unhealthy",   "color":"#a855f7","bg":"#f3e8ff",
                "desc":"Health alert — everyone may experience serious health effects."}
    return {"label":"Hazardous",            "color":"#7f1d1d","bg":"#fecaca",
            "desc":"Health warning of emergency conditions. Entire population is likely affected."}


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

@app.route("/")
def home():
    stats = _cached("dash_stats", real_dashboard_stats)
    return render_template("home.html", stats=stats)


@app.route("/dashboard")
def dashboard():
    stats  = _cached("dash_stats", real_dashboard_stats)
    images = list_output_images()
    # Pick one representative chart per category for the dashboard preview
    preview_cats = ["Correlation","Trend","Histogram","Missing Values"]
    previews = []
    for cat in preview_cats:
        for img in images:
            if img["category"] == cat:
                previews.append(img)
                break
    corr = load_corr_top()
    return render_template("dashboard.html",
                           stats=stats,
                           previews=previews,
                           corr=corr)


@app.route("/dataset")
def dataset():
    files    = list_dataset_files()
    selected = request.args.get("file",
                files[0]["name"] if files else "")
    page     = max(1, int(request.args.get("page", 1)))
    search   = request.args.get("q", "").strip()
    per_page = 50

    df        = load_dataframe_safe(selected) if selected else None
    rows      = []
    columns   = []
    total_rows   = 0
    total_cols   = 0
    missing_total = 0
    numeric_count = 0
    col_stats    = []
    total_pages  = 1

    if df is not None:
        columns       = df.columns.tolist()
        total_rows    = len(df)
        total_cols    = len(columns)
        missing_total = int(df.isnull().sum().sum())
        numeric_count = len(df.select_dtypes(include="number").columns)
        col_stats     = get_col_stats(df)
        view = df
        if search:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            view = df[mask]
        total_pages  = max(1, (len(view) + per_page - 1) // per_page)
        page         = min(page, total_pages)
        start        = (page - 1) * per_page
        rows         = view.iloc[start:start+per_page].fillna("—").values.tolist()

    return render_template(
        "dataset.html",
        dataset_files = files,
        selected_file = selected,
        columns       = columns,
        rows          = rows,
        total_rows    = f"{total_rows:,}",
        total_cols    = total_cols,
        missing_total = f"{missing_total:,}",
        numeric_count = numeric_count,
        col_stats     = col_stats,
        page          = page,
        total_pages   = total_pages,
        search        = search,
        per_page      = per_page,
    )


@app.route("/visualization")
def visualization():
    images  = list_output_images()
    grouped = categorise_images(images)
    cats    = list(grouped.keys())
    sel_cat = request.args.get("cat", "All")
    search  = request.args.get("q", "").strip().lower()
    if sel_cat not in grouped:
        sel_cat = "All"
    shown = grouped[sel_cat]
    if search:
        shown = [i for i in shown
                 if search in i["label"].lower() or search in i["filename"].lower()]
    cat_counts = {c: len(grouped[c]) for c in cats}
    return render_template(
        "visualization.html",
        categories  = cats,
        selected_cat= sel_cat,
        images      = shown,
        search      = search,
        cat_counts  = cat_counts,
        total_charts= len(grouped.get("All", [])),
    )


@app.route("/preprocessing")
def preprocessing():
    # Check which cleaned datasets actually exist
    cleaned = {}
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".csv") and fname != "PRSA_Data_Aotizhongxin_raw.csv":
            fpath = os.path.join(DATASET_DIR, fname)
            try:
                df = pd.read_csv(fpath, nrows=1)
                nrows = sum(1 for _ in open(fpath)) - 1
                cleaned[fname] = {"rows": nrows, "cols": len(df.columns)}
            except Exception:
                cleaned[fname] = {"rows": 0, "cols": 0}
    raw_stats = _cached("dash_stats", real_dashboard_stats)
    return render_template("preprocessing.html",
                           cleaned=cleaned,
                           raw_stats=raw_stats)


@app.route("/models")
def models():
    model_results = load_model_results()
    return render_template("models.html", model_results=model_results)


@app.route("/model/<key>")
def model_detail(key):
    # Validate key against alphanumeric + underscore only
    if not re.fullmatch(r"[a-z0-9_]+", key):
        abort(400)
    model_results = load_model_results()
    m = model_results.get(key)
    if not m:
        return redirect(url_for("models"))
    return render_template("model_detail.html", key=key, m=m)


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_val = None
    aqi_info       = None
    form_data      = {}
    error          = None
    model_ready    = (_TRAINED_MODEL is not None)

    if request.method == "POST":
        form_data = {k: v.strip() for k, v in request.form.items()}
        prediction_val, error = predict_pm25(form_data)
        if prediction_val is not None:
            aqi_info = aqi_category(prediction_val)

    return render_template(
        "prediction.html",
        prediction  = prediction_val,
        aqi_info    = aqi_info,
        form_data   = form_data,
        error       = error,
        model_ready = model_ready,
        features    = _MODEL_FEATURES,
    )


@app.route("/reports")
def reports():
    return render_template(
        "reports.html",
        csv_reports    = list_csv_reports(),
        missing_report = load_missing_report(),
        corr_report    = load_corr_top(),
        stat_summary   = load_stat_summary(),
    )


@app.route("/about")
def about():
    stats = _cached("dash_stats", real_dashboard_stats)
    return render_template("about.html", stats=stats)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    sent = False
    # No email backend — only show success UI, do not claim delivery
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if name and email and message:
            sent = True
    return render_template("contact.html", sent=sent)


# ─────────────────────────────────────────────────────────────
# Static file routes
# ─────────────────────────────────────────────────────────────

@app.route("/output/<path:filename>")
def output_file(filename):
    safe = os.path.basename(filename)          # prevent path traversal
    full = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(full):
        abort(404)
    return send_from_directory(OUTPUT_DIR, safe)


@app.route("/download/<path:filename>")
def download_report(filename):
    safe = os.path.basename(filename)
    full = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(full):
        abort(404)
    return send_from_directory(OUTPUT_DIR, safe, as_attachment=True)


@app.route("/download/dataset/<path:filename>")
def download_dataset(filename):
    safe = os.path.basename(filename)
    full = os.path.join(DATASET_DIR, safe)
    if not os.path.exists(full):
        abort(404)
    return send_from_directory(DATASET_DIR, safe, as_attachment=True)


# ─────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404,
                           message="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    log.error("500: %s", e)
    return render_template("error.html", code=500,
                           message="Internal server error."), 500

@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", code=400,
                           message="Bad request."), 400


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AERO-PURE  Air Quality ML Dashboard  v" + APP_VERSION)
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, use_reloader=False)
