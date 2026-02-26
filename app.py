"""
=============================================================
 Education ML API — Flask Deployment
=============================================================
 Endpoints:
   POST /predict/dropout      → Risk score + alert flag
   POST /predict/score        → Predicted reading + math
   POST /cluster/assign       → Assign student to risk cluster
   POST /simulate/policy      → Counterfactual policy impact
=============================================================
 Run:
   pip install flask joblib scikit-learn pandas numpy
   python app.py
=============================================================
"""

from flask import Flask, request, jsonify
import joblib, json
import numpy as np
import pandas as pd
from functools import wraps

app = Flask(__name__)

# ── Load models ─────────────────────────────────────────────────
MODEL_DIR = "./models"
model_dropout  = joblib.load(f"{MODEL_DIR}/model1_dropout.pkl")
model_score_r  = joblib.load(f"{MODEL_DIR}/model2_score_reading.pkl")
model_score_m  = joblib.load(f"{MODEL_DIR}/model2_score_math.pkl")
model_cluster  = joblib.load(f"{MODEL_DIR}/model3_cluster.pkl")
scaler_cluster = joblib.load(f"{MODEL_DIR}/scaler_cluster.pkl")

with open(f"{MODEL_DIR}/model_metadata.json", "r") as f:
    META = json.load(f)

THRESHOLD    = META["model1"]["optimal_threshold"]
FEAT_DROPOUT = META["model1"]["features"]
FEAT_SCORE   = META["model2"]["features"]
FEAT_CLUSTER = META["model3"]["features"]

CLUSTER_RISK = {
    int(k): v for k, v in META["model3"]["cluster_risk_labels"].items()
}

RISK_COLORS  = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}

# ── Helpers ─────────────────────────────────────────────────────
def validate_input(data: dict, required_fields: list) -> tuple[bool, str]:
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, ""

def enrich_features(d: dict) -> dict:
    """Derived features (must mirror training pipeline)"""
    d["digital_access"]   = int(d.get("internet_home", 0) * d.get("device_access", 0))
    d["policy_support_n"] = sum([
        d.get("scholarship", 0), d.get("free_lunch", 0),
        d.get("device_subsidy", 0), d.get("internet_subsidy", 0),
        d.get("remedial_program", 0),
    ])
    d["vulnerable_score"] = (
        int(d.get("ses_quintile", 3) <= 2) +
        d.get("disability", 0) + d.get("ethnicity_minority", 0) +
        d.get("migrant_status", 0)
    )
    d["low_attendance"]   = int(d.get("attendance_rate", 1.0) < 0.75)
    d["remote_no_internet"] = int(
        d.get("distance_km", 0) > 10 and d.get("internet_home", 0) == 0
    )
    d["sex_enc"]          = 1 if d.get("sex", "M") == "M" else 0
    d["avg_baseline"]     = (
        d.get("baseline_skill_reading", 50) + d.get("baseline_skill_math", 50)
    ) / 2
    d["avg_gain"]         = (
        d.get("learning_gain_reading", 0) + d.get("learning_gain_math", 0)
    ) / 2
    d["score_gap"]        = (
        d.get("score_reading", 50) - d.get("score_math", 50)
    )
    d["is_vocational"]    = 1 if str(d.get("grade_code", "G1")).startswith("V") else 0
    d["grade_num"]        = int('''
        str(d.get("grade_code","G1")).replace("G","").replace("V","")
    '''.strip()) if d.get("grade_code") else 1
    d["year_centered"]    = int(d.get("academic_year", 2562)) - 2562
    d["covid_year"]       = 1 if int(d.get("academic_year", 2562)) in [2563, 2564] else 0
    return d

# ── Endpoint 1: Dropout Risk ─────────────────────────────────────
@app.route("/predict/dropout", methods=["POST"])
def predict_dropout():
    data = request.json
    ok, msg = validate_input(data, ["ses_quintile", "attendance_rate", "distance_km"])
    if not ok:
        return jsonify({"error": msg}), 400

    data = enrich_features(data)
    row  = pd.DataFrame([data]).reindex(columns=FEAT_DROPOUT, fill_value=0)

    prob     = float(model_dropout.predict_proba(row)[0, 1])
    is_at_risk = prob >= THRESHOLD
    risk_level = ("High" if prob >= 0.5 else
                  "Medium" if prob >= THRESHOLD else "Low")

    return jsonify({
        "student_id"       : data.get("student_id", "unknown"),
        "dropout_probability": round(prob, 4),
        "at_risk_flag"     : bool(is_at_risk),
        "risk_level"       : risk_level,
        "threshold_used"   : THRESHOLD,
        "recommended_action": (
            "🔴 Immediate intervention: counseling + scholarship review"
            if risk_level == "High" else
            "🟡 Monitor closely: attendance follow-up"
            if risk_level == "Medium" else
            "🟢 On track — routine check-in"
        ),
    })

# ── Endpoint 2: Score Prediction ─────────────────────────────────
@app.route("/predict/score", methods=["POST"])
def predict_score():
    data = request.json
    data = enrich_features(data)

    row_r = pd.DataFrame([data]).reindex(columns=FEAT_SCORE, fill_value=0)
    row_m = pd.DataFrame([data]).reindex(columns=FEAT_SCORE, fill_value=0)

    pred_r = float(np.clip(model_score_r.predict(row_r)[0], 0, 100))
    pred_m = float(np.clip(model_score_m.predict(row_m)[0], 0, 100))

    return jsonify({
        "student_id"             : data.get("student_id", "unknown"),
        "predicted_score_reading": round(pred_r, 2),
        "predicted_score_math"   : round(pred_m, 2),
        "predicted_avg_score"    : round((pred_r + pred_m) / 2, 2),
        "performance_band"       : (
            "Below Basic" if (pred_r + pred_m)/2 < 40 else
            "Basic"       if (pred_r + pred_m)/2 < 55 else
            "Proficient"  if (pred_r + pred_m)/2 < 70 else
            "Advanced"
        ),
    })

# ── Endpoint 3: Cluster Assignment ───────────────────────────────
@app.route("/cluster/assign", methods=["POST"])
def cluster_assign():
    data = request.json
    data = enrich_features(data)

    row    = pd.DataFrame([data]).reindex(columns=FEAT_CLUSTER, fill_value=0)
    row_sc = scaler_cluster.transform(row)
    cluster_id = int(model_cluster.predict(row_sc)[0])
    risk_label = CLUSTER_RISK.get(cluster_id, "Unknown")

    return jsonify({
        "student_id" : data.get("student_id", "unknown"),
        "cluster_id" : cluster_id,
        "risk_label" : risk_label,
        "risk_icon"  : RISK_COLORS.get(risk_label, "⚪"),
    })

# ── Endpoint 4: Policy Simulation ────────────────────────────────
@app.route("/simulate/policy", methods=["POST"])
def simulate_policy():
    """
    ส่ง student data + list of policies ที่จะ simulate
    คืนค่า dropout probability before/after + score change
    """
    data     = request.json
    policies = data.pop("policies", [])   # e.g. ["scholarship","internet","remedial"]

    data = enrich_features(data)
    row_before = pd.DataFrame([data]).reindex(columns=FEAT_DROPOUT, fill_value=0)
    prob_before = float(model_dropout.predict_proba(row_before)[0, 1])

    # apply policies
    data_after = data.copy()
    for policy in policies:
        if policy == "scholarship"  : data_after["scholarship"] = 1
        if policy == "internet"     : data_after.update({"internet_home":1,"internet_subsidy":1,"digital_access":1})
        if policy == "device"       : data_after.update({"device_access":1,"device_subsidy":1,"digital_access":1})
        if policy == "remedial"     : data_after["remedial_program"] = 1
        if policy == "free_lunch"   : data_after["free_lunch"] = 1
    data_after = enrich_features(data_after)

    row_after  = pd.DataFrame([data_after]).reindex(columns=FEAT_DROPOUT, fill_value=0)
    prob_after = float(model_dropout.predict_proba(row_after)[0, 1])

    row_score_b = pd.DataFrame([data]).reindex(columns=FEAT_SCORE, fill_value=0)
    row_score_a = pd.DataFrame([data_after]).reindex(columns=FEAT_SCORE, fill_value=0)
    score_b = float(model_score_r.predict(row_score_b)[0])
    score_a = float(model_score_r.predict(row_score_a)[0])

    return jsonify({
        "policies_applied"    : policies,
        "dropout_risk_before" : round(prob_before, 4),
        "dropout_risk_after"  : round(prob_after, 4),
        "dropout_risk_change" : round(prob_after - prob_before, 4),
        "reading_score_before": round(score_b, 2),
        "reading_score_after" : round(score_a, 2),
        "reading_score_change": round(score_a - score_b, 2),
    })

# ── Health check ─────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": ["dropout", "score_reading", "score_math", "cluster"],
        "model1_auc"  : META["model1"]["auc"],
        "model2_mae_r": META["model2"]["mae_reading"],
        "model3_k"    : META["model3"]["k"],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
