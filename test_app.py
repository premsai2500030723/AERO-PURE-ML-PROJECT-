"""
test_app.py — AERO-PURE full test suite
Run:  python test_app.py
"""
import os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__))

PASS = 0; FAIL = 0

def ok(name):
    global PASS; PASS += 1; print(f"  ✅  {name}")

def fail(name, reason=""):
    global FAIL; FAIL += 1; print(f"  ❌  {name}" + (f" — {reason}" if reason else ""))

print("\n" + "="*60)
print("  AERO-PURE Test Suite")
print("="*60)

# ── 1. App startup ────────────────────────────────────────────
print("\n── 1. App startup ──────────────────────────────────────")
try:
    from app import app, load_raw, list_output_images, load_model_results
    app.config["TESTING"] = True
    client = app.test_client()
    ok("App imports without errors")
except Exception as e:
    fail("App import", str(e)); sys.exit(1)

# ── 2. All routes ─────────────────────────────────────────────
print("\n── 2. All routes (GET) ─────────────────────────────────")
routes = ["/","/dashboard","/dataset","/visualization",
          "/preprocessing","/models","/prediction","/reports","/about","/contact"]
for route in routes:
    r = client.get(route)
    if r.status_code == 200:
        ok(f"GET {route}")
    else:
        fail(f"GET {route}", f"HTTP {r.status_code}")

# Model detail pages
print("\n── 3. Model detail pages ───────────────────────────────")
model_results = load_model_results()
for key in model_results:
    r = client.get(f"/model/{key}")
    if r.status_code == 200:
        ok(f"GET /model/{key}")
    else:
        fail(f"GET /model/{key}", f"HTTP {r.status_code}")

# Bad model key — should redirect
r = client.get("/model/nonexistent_key_xyz")
if r.status_code in (302, 404):
    ok("GET /model/invalid_key → redirect or 404")
else:
    fail("GET /model/invalid_key", f"HTTP {r.status_code}")

# ── 4. Dataset loading ─────────────────────────────────────────
print("\n── 4. Dataset loading ──────────────────────────────────")
df = load_raw()
if df is not None and len(df) > 0:
    ok(f"Raw dataset loaded — {len(df):,} rows × {len(df.columns)} cols")
else:
    fail("Raw dataset load")

import pandas as pd
for fname in ["PRSA_Data_Aotizhongxin_raw.csv","clean_label_encode_air_quality.csv"]:
    fpath = os.path.join(BASE,"dataset",fname)
    if os.path.exists(fpath):
        ok(f"Dataset file exists: {fname}")
    else:
        fail(f"Dataset file missing: {fname}")

# ── 5. Real stats ─────────────────────────────────────────────
print("\n── 5. Real dashboard stats ─────────────────────────────")
from app import real_dashboard_stats
stats = real_dashboard_stats()
for key, expected in [("total_rows","35,064"),("total_cols",18),("missing","7,271")]:
    val = str(stats.get(key,""))
    if str(expected) in val or val == str(expected):
        ok(f"stats['{key}'] = {val}")
    else:
        fail(f"stats['{key}']", f"got {val!r}, expected {expected!r}")

# ── 6. Chart discovery ─────────────────────────────────────────
print("\n── 6. Chart discovery ──────────────────────────────────")
images = list_output_images()
if len(images) >= 50:
    ok(f"Charts discovered: {len(images)}")
else:
    fail("Chart count", f"only {len(images)} found")

# Verify No_* charts are excluded
no_charts = [i for i in images if i["filename"].lower().startswith("no_")]
if len(no_charts) == 0:
    ok("No_* charts correctly excluded")
else:
    fail("No_* charts not excluded", str(no_charts))

# Every chart must have category
uncategorised = [i for i in images if not i.get("category")]
if not uncategorised:
    ok("All charts have categories")
else:
    fail("Uncategorised charts", str([i["filename"] for i in uncategorised]))

# ── 7. Chart serving ───────────────────────────────────────────
print("\n── 7. Chart serving ────────────────────────────────────")
test_charts = ["Correlation_Heatmap.png","Missing_Values_Heatmap.png","Pollution_PairPlot.png"]
for fname in test_charts:
    fpath = os.path.join(BASE,"output",fname)
    if os.path.exists(fpath):
        r = client.get(f"/output/{fname}")
        if r.status_code == 200:
            ok(f"Served: {fname}")
        else:
            fail(f"Serve {fname}", f"HTTP {r.status_code}")
    else:
        fail(f"File missing: {fname}")

# Path traversal protection
r = client.get("/output/../app.py")
if r.status_code in (404, 400):
    ok("Path traversal blocked on /output/")
else:
    fail("Path traversal not blocked", f"HTTP {r.status_code}")

# ── 8. Reports ────────────────────────────────────────────────
print("\n── 8. Reports ──────────────────────────────────────────")
for fname in ["Missing_Values_Report.csv","Correlation_Matrix.csv","Statistical_Summary.csv"]:
    fpath = os.path.join(BASE,"output",fname)
    if os.path.exists(fpath):
        ok(f"Report exists: {fname}")
        r = client.get(f"/download/{fname}")
        if r.status_code == 200:
            ok(f"Download works: {fname}")
        else:
            fail(f"Download {fname}", f"HTTP {r.status_code}")
    else:
        fail(f"Report missing: {fname}")

# ── 9. Model metrics ──────────────────────────────────────────
print("\n── 9. Model results JSON ───────────────────────────────")
if model_results:
    ok(f"model_results.json loaded — {len(model_results)} models")
    rf = model_results.get("random_forest",{})
    if rf.get("r2",0) >= 0.90:
        ok(f"Random Forest R² = {rf['r2']} (≥0.90)")
    else:
        fail("Random Forest R²", f"got {rf.get('r2')}")
    lr = model_results.get("logistic_binary",{})
    if lr.get("accuracy",0) >= 0.85:
        ok(f"Logistic Binary Acc = {lr['accuracy']} (≥0.85)")
    else:
        fail("Logistic Binary accuracy", f"got {lr.get('accuracy')}")
else:
    fail("model_results.json empty or missing")

# ── 10. Prediction ────────────────────────────────────────────
print("\n── 10. Prediction ──────────────────────────────────────")
model_path = os.path.join(BASE,"models","pm25_model.pkl")
if os.path.exists(model_path):
    ok("Trained model file exists: models/pm25_model.pkl")
else:
    fail("Trained model missing — run train_model.py")

# Valid prediction
r = client.post("/prediction", data={
    "pm10":"148","so2":"22","no2":"68","co":"1200",
    "o3":"12","temp":"-3","pres":"1020","dewp":"-10","rain":"0","wspm":"1.2"
})
if r.status_code == 200:
    body = r.data.decode()
    if "µg/m³" in body or "Predicted" in body:
        ok("POST /prediction returns PM2.5 result")
    else:
        fail("POST /prediction — no result in response")
else:
    fail("POST /prediction", f"HTTP {r.status_code}")

# Missing field
r = client.post("/prediction", data={"pm10":"100"})
if r.status_code == 200:
    body = r.data.decode()
    if "Missing" in body or "error" in body.lower() or "Invalid" in body:
        ok("Incomplete form shows error message")
    else:
        fail("Incomplete form — no error shown")
else:
    fail("Incomplete POST /prediction", f"HTTP {r.status_code}")

# ── 11. Dataset download ──────────────────────────────────────
print("\n── 11. Dataset downloads ───────────────────────────────")
r = client.get("/download/dataset/PRSA_Data_Aotizhongxin_raw.csv")
if r.status_code == 200:
    ok("Download raw dataset works")
else:
    fail("Download raw dataset", f"HTTP {r.status_code}")

# ── 12. Error handlers ────────────────────────────────────────
print("\n── 12. Error handlers ──────────────────────────────────")
r = client.get("/this_page_does_not_exist_xyz")
if r.status_code == 404:
    ok("404 handler works")
else:
    fail("404 handler", f"HTTP {r.status_code}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed  |  {FAIL} failed")
print("="*60)
if FAIL == 0:
    print("  🎉 All tests passed!")
else:
    print("  ⚠️  Some tests failed — see above.")
print()
sys.exit(0 if FAIL == 0 else 1)
