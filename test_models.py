"""
test_models.py — AERO-PURE model page tests
Run: python test_models.py
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).parent
DATA = json.loads((ROOT/"output"/"model_report"/"models.json").read_text())
INDEX = {m["id"]: m for m in DATA}
TEST_N = 6828   # 20% of 34,139

PASS = FAIL = 0

def ok(msg):
    global PASS; PASS += 1; print(f"  OK   {msg}")

def fail(msg, detail=""):
    global FAIL; FAIL += 1; print(f"  FAIL {msg}" + (f" -- {detail}" if detail else ""))

print("\n" + "="*60)
print("  model_report tests")
print("="*60)

# 1. /models route
from app import app
app.config["TESTING"] = True
c = app.test_client()
r = c.get("/models")
if r.status_code == 200: ok("/models returns 200")
else: fail("/models", f"HTTP {r.status_code}")

# 2. Every model id returns 200 on detail route
for mid in INDEX:
    r = c.get(f"/model/{mid}")
    if r.status_code == 200: ok(f"/model/{mid} returns 200")
    else: fail(f"/model/{mid}", f"HTTP {r.status_code}")

# 3. /api/models returns valid JSON list
r = c.get("/api/models")
if r.status_code == 200 and r.content_type.startswith("application/json"):
    ok("/api/models returns JSON")
else:
    fail("/api/models", f"{r.status_code} {r.content_type}")

# 4. Accuracy values between 0 and 1
for m in DATA:
    for section in ["binary", "multiclass"]:
        s = m.get(section, {})
        acc = s.get("accuracy")
        if acc is not None:
            if 0 <= acc <= 1:
                ok(f"{m['id']}.{section}.accuracy in [0,1]: {acc}")
            else:
                fail(f"{m['id']}.{section}.accuracy out of range", str(acc))

# 5. Every confusion matrix sums to TEST_N
for m in DATA:
    for section in ["binary", "multiclass"]:
        s = m.get(section, {})
        cm = s.get("matrix")
        if cm:
            total = sum(sum(row) for row in cm)
            if total == TEST_N:
                ok(f"{m['id']}.{section} CM sums to {TEST_N}")
            else:
                fail(f"{m['id']}.{section} CM sum wrong", f"{total} != {TEST_N}")

# 6. Regression metrics finite for regressors
for m in DATA:
    reg = m.get("regression")
    if reg:
        for k in ["r2", "mae", "mse", "rmse"]:
            v = reg.get(k)
            if v is not None and -1 < v < 1e9:
                ok(f"{m['id']}.regression.{k} finite: {v}")
            else:
                fail(f"{m['id']}.regression.{k}", str(v))

# 7. PNG files exist
for m in DATA:
    for section in ["binary", "multiclass"]:
        s = m.get(section, {})
        png = s.get("confusion_matrix_png")
        if png:
            path = ROOT/"output"/"model_report"/png
            if path.exists():
                ok(f"PNG exists: {png}")
            else:
                fail(f"PNG missing: {png}")
    roc = m.get("roc_png")
    if roc:
        path = ROOT/"output"/"model_report"/roc
        if path.exists(): ok(f"ROC PNG exists: {roc}")
        else: fail(f"ROC PNG missing: {roc}")

print()
print("="*60)
print(f"  {PASS}/{PASS+FAIL} passed  |  {FAIL} failed")
print("="*60)
print("  All tests passed!" if FAIL==0 else "  Some failures above.")
sys.exit(0 if FAIL == 0 else 1)
