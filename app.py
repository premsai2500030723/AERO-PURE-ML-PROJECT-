"""
app.py  –  Flask web application for the Air Quality ML Project
Run:  python app.py
"""

import os
import math
import pandas as pd
from flask import (
    Flask, render_template, request,
    send_from_directory, redirect, url_for
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

APP_VERSION = "1.0.0"

app = Flask(__name__)
app.secret_key = "aq_ml_secret_2026"
app.jinja_env.globals.update(zip=zip, enumerate=enumerate)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def list_dataset_files():
    """Return CSV filenames present in the dataset folder."""
    return [
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".csv")
    ]


def load_dataframe(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_col_stats(df: pd.DataFrame) -> list[dict]:
    stats = []
    total = len(df)
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        pct = round(missing / total * 100, 2) if total else 0
        stats.append({
            "name":        col,
            "dtype":       str(df[col].dtype),
            "missing":     missing,
            "missing_pct": pct,
            "unique":      int(df[col].nunique()),
        })
    return stats


def list_output_images() -> list[str]:
    """Return all PNG filenames from the output folder."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".png")
    )


def categorise_images(images: list[str]) -> dict[str, list[dict]]:
    """Group images by type based on filename suffix."""
    groups: dict[str, list[dict]] = {
        "All":         [],
        "Histogram":   [],
        "Boxplot":     [],
        "Outlier":     [],
        "Heatmap":     [],
        "Scatter":     [],
        "Count Plot":  [],
        "Pair Plot":   [],
        "Other":       [],
    }
    for fname in images:
        label = fname.replace("_", " ").replace(".png", "")
        item = {"filename": fname, "label": label}
        groups["All"].append(item)
        lower = fname.lower()
        if "histogram" in lower:
            groups["Histogram"].append(item)
        elif "boxplot" in lower:
            groups["Boxplot"].append(item)
        elif "outlier" in lower:
            groups["Outlier"].append(item)
        elif "heatmap" in lower:
            groups["Heatmap"].append(item)
        elif "scatter" in lower:
            groups["Scatter"].append(item)
        elif "countplot" in lower or "count" in lower:
            groups["Count Plot"].append(item)
        elif "pairplot" in lower or "pair" in lower:
            groups["Pair Plot"].append(item)
        else:
            groups["Other"].append(item)
    # Remove empty categories (except All)
    return {k: v for k, v in groups.items() if k == "All" or v}


def get_aqi_category(pm25: float) -> str:
    if pm25 <= 12:
        return "Good 🟢"
    elif pm25 <= 35:
        return "Moderate 🟡"
    elif pm25 <= 55:
        return "Unhealthy for Sensitive Groups 🟠"
    elif pm25 <= 150:
        return "Unhealthy 🔴"
    elif pm25 <= 250:
        return "Very Unhealthy 🟣"
    else:
        return "Hazardous ⚫"


def simple_predict_pm25(form: dict) -> float:
    """
    Simple heuristic prediction (placeholder until a real model is trained).
    Uses weighted sum of correlated features.
    """
    pm10  = float(form.get("pm10",  0) or 0)
    co    = float(form.get("co",    0) or 0)
    no2   = float(form.get("no2",   0) or 0)
    temp  = float(form.get("temp",  15) or 15)
    wspm  = float(form.get("wspm",  2) or 2)
    rain  = float(form.get("rain",  0) or 0)

    # Rough coefficients inspired by correlation analysis
    pred = (
        0.40 * pm10
        + 0.015 * co
        + 0.25 * no2
        - 0.30 * temp
        - 1.50 * wspm
        - 0.80 * rain
        + 12.0          # intercept
    )
    return round(max(pred, 0.0), 1)


def load_missing_report() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "Missing_Values_Report.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        return []


def load_corr_top(n: int = 12) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "Correlation_Matrix.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, index_col=0)
        if "PM2.5" not in df.columns:
            return []
        corr = df["PM2.5"].drop("PM2.5", errors="ignore").sort_values(
            key=abs, ascending=False
        ).head(n)
        return [{"feature": k, "value": round(v, 4)} for k, v in corr.items()]
    except Exception:
        return []


def list_csv_reports() -> list[dict]:
    reports = []
    candidates = {
        "Missing_Values_Report.csv": "Missing values count and percentage per column",
        "Correlation_Matrix.csv":    "Full pairwise correlation matrix for numeric columns",
        "Statistical_Summary.csv":   "Descriptive statistics (mean, std, min, max, quartiles)",
    }
    for fname, desc in candidates.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            reports.append({
                "name":     fname.replace("_", " ").replace(".csv", ""),
                "filename": fname,
                "desc":     desc,
                "size":     f"{size_kb} KB",
            })
    return reports


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    # Try to load stats from the raw dataset
    raw_file = "PRSA_Data_Aotizhongxin_raw.csv"
    df = load_dataframe(raw_file)
    stats = {}
    if df is not None:
        stats = {
            "total_rows":   f"{len(df):,}",
            "total_cols":   len(df.columns),
            "missing":      f"{int(df.isnull().sum().sum()):,}",
            "numeric_cols": len(df.select_dtypes(include="number").columns),
            "cat_cols":     len(df.select_dtypes(include="object").columns),
            "charts":       len(list_output_images()),
        }
    return render_template("dashboard.html", stats=stats)


@app.route("/dataset")
def dataset():
    files        = list_dataset_files()
    selected     = request.args.get("file", files[0] if files else "")
    df           = load_dataframe(selected) if selected else None

    rows         = []
    columns      = []
    total_rows   = 0
    total_cols   = 0
    missing_total = 0
    numeric_count = 0
    col_stats    = []

    if df is not None:
        columns       = df.columns.tolist()
        total_rows    = len(df)
        total_cols    = len(columns)
        missing_total = int(df.isnull().sum().sum())
        numeric_count = len(df.select_dtypes(include="number").columns)
        col_stats     = get_col_stats(df)
        # Show first 100 rows
        preview = df.head(100)
        rows = preview.values.tolist()

    return render_template(
        "dataset.html",
        dataset_files  = files,
        selected_file  = selected,
        columns        = columns,
        rows           = rows,
        total_rows     = f"{total_rows:,}",
        total_cols     = total_cols,
        missing_total  = f"{missing_total:,}",
        numeric_count  = numeric_count,
        col_stats      = col_stats,
    )


@app.route("/visualization")
def visualization():
    all_images = list_output_images()
    grouped    = categorise_images(all_images)
    categories = list(grouped.keys())

    selected_cat = request.args.get("cat", "All")
    if selected_cat not in grouped:
        selected_cat = "All"

    images = grouped.get(selected_cat, [])
    return render_template(
        "visualization.html",
        categories   = categories,
        selected_cat = selected_cat,
        images       = images,
    )


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    pred_value   = None
    aqi_category = None
    form_data    = {}

    if request.method == "POST":
        form_data    = request.form.to_dict()
        pred_value   = simple_predict_pm25(form_data)
        aqi_category = get_aqi_category(pred_value)

    return render_template(
        "prediction.html",
        prediction   = pred_value,
        aqi_category = aqi_category,
        form_data    = form_data,
    )


@app.route("/reports")
def reports():
    return render_template(
        "reports.html",
        csv_reports    = list_csv_reports(),
        missing_report = load_missing_report(),
        corr_report    = load_corr_top(),
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    sent = False
    if request.method == "POST":
        # In a real app you would send an email or store the message.
        # For the prototype we just set a flag.
        sent = True
    return render_template("contact.html", sent=sent)


# ─────────────────────────────────────────────────────────────
# Static file helpers
# ─────────────────────────────────────────────────────────────

@app.route("/output/<path:filename>")
def output_file(filename):
    """Serve images from the output/ folder."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/download/<path:filename>")
def download_report(filename):
    """Download a CSV report from the output/ folder."""
    return send_from_directory(
        OUTPUT_DIR, filename, as_attachment=True
    )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Air Quality ML Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)

