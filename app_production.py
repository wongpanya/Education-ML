import os
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except Exception:  # optional dependency
    CORS = None


# ---------------- App / Logging ----------------
app = Flask(__name__)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("education-ml-api")

if CORS is not None:
    CORS(app)
    logger.info("CORS enabled")
else:
    logger.info("flask-cors not installed; CORS disabled")


# ---------------- Model loading ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

META = {}
THRESHOLD = 0.5
FEAT_DROPOUT = []
FEAT_SCORE = []
FEAT_CLUSTER = []
CLUSTER_RISK = {}
RISK_COLORS = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}

model_dropout = None
model_score_r = None
model_score_m = None
model_cluster = None
scaler_cluster = None


def _load_artifacts():
    global META, THRESHOLD, FEAT_DROPOUT, FEAT_SCORE, FEAT_CLUSTER, CLUSTER_RISK
    global model_dropout, model_score_r, model_score_m, model_cluster, scaler_cluster

    logger.info("Loading models from %s", MODEL_DIR)
    model_dropout = joblib.load(MODEL_DIR / "model1_dropout.pkl")
    model_score_r = joblib.load(MODEL_DIR / "model2_score_reading.pkl")
    model_score_m = joblib.load(MODEL_DIR / "model2_score_math.pkl")
    model_cluster = joblib.load(MODEL_DIR / "model3_cluster.pkl")
    scaler_cluster = joblib.load(MODEL_DIR / "scaler_cluster.pkl")

    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        META = json.load(f)

    THRESHOLD = float(META["model1"]["optimal_threshold"])
    FEAT_DROPOUT = META["model1"]["features"]
    FEAT_SCORE = META["model2"]["features"]
    FEAT_CLUSTER = META["model3"]["features"]
    CLUSTER_RISK = {int(k): v for k, v in META["model3"]["cluster_risk_labels"].items()}


MODEL_LOAD_ERROR = None
try:
    _load_artifacts()
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    logger.exception("Model loading failed")


# ---------------- Utilities ----------------
def api_error(message: str, status: int = 400, **extra):
    payload = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def get_json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be JSON object")
    return data


def validate_input(data: dict, required_fields: list) -> tuple[bool, str]:
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, ""


def enrich_features(d: dict) -> dict:
    d = dict(d)  # avoid mutating original input unexpectedly
    d["digital_access"] = int(d.get("internet_home", 0) * d.get("device_access", 0))
    d["policy_support_n"] = sum(
        [
            d.get("scholarship", 0),
            d.get("free_lunch", 0),
            d.get("device_subsidy", 0),
            d.get("internet_subsidy", 0),
            d.get("remedial_program", 0),
        ]
    )
    d["vulnerable_score"] = (
        int(d.get("ses_quintile", 3) <= 2)
        + d.get("disability", 0)
        + d.get("ethnicity_minority", 0)
        + d.get("migrant_status", 0)
    )
    d["low_attendance"] = int(float(d.get("attendance_rate", 1.0)) < 0.75)
    d["remote_no_internet"] = int(
        float(d.get("distance_km", 0)) > 10 and int(d.get("internet_home", 0)) == 0
    )
    d["sex_enc"] = 1 if str(d.get("sex", "M")).upper() == "M" else 0
    d["avg_baseline"] = (
        float(d.get("baseline_skill_reading", 50)) + float(d.get("baseline_skill_math", 50))
    ) / 2
    d["avg_gain"] = (
        float(d.get("learning_gain_reading", 0)) + float(d.get("learning_gain_math", 0))
    ) / 2
    d["score_gap"] = float(d.get("score_reading", 50)) - float(d.get("score_math", 50))

    grade_code = str(d.get("grade_code", "G1"))
    d["is_vocational"] = 1 if grade_code.startswith("V") else 0
    grade_digits = grade_code.replace("G", "").replace("V", "")
    d["grade_num"] = int(grade_digits) if grade_digits.isdigit() else 1

    academic_year = int(d.get("academic_year", 2562))
    d["year_centered"] = academic_year - 2562
    d["covid_year"] = 1 if academic_year in [2563, 2564] else 0
    return d


def ensure_models_ready():
    if MODEL_LOAD_ERROR is not None:
        raise RuntimeError(f"Models not loaded: {MODEL_LOAD_ERROR}")


# ---------------- Error handlers ----------------
@app.errorhandler(400)
def handle_400(e):
    return api_error("Bad request", 400, detail=str(e))


@app.errorhandler(404)
def handle_404(e):
    return api_error("Not found", 404)


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled error on %s", request.path)
    return api_error("Internal server error", 500, detail=str(e))


@app.before_request
def log_request_summary():
    logger.info("%s %s", request.method, request.path)


# ---------------- Endpoints ----------------
@app.route("/predict/dropout", methods=["POST"])
def predict_dropout():
    ensure_models_ready()
    data = get_json_body()
    ok, msg = validate_input(data, ["ses_quintile", "attendance_rate", "distance_km"])
    if not ok:
        return api_error(msg, 400)

    data = enrich_features(data)
    row = pd.DataFrame([data]).reindex(columns=FEAT_DROPOUT, fill_value=0)

    prob = float(model_dropout.predict_proba(row)[0, 1])
    is_at_risk = prob >= THRESHOLD
    risk_level = "High" if prob >= 0.5 else "Medium" if prob >= THRESHOLD else "Low"

    return jsonify(
        {
            "student_id": data.get("student_id", "unknown"),
            "dropout_probability": round(prob, 4),
            "at_risk_flag": bool(is_at_risk),
            "risk_level": risk_level,
            "threshold_used": THRESHOLD,
            "recommended_action": (
                "🔴 Immediate intervention: counseling + scholarship review"
                if risk_level == "High"
                else "🟡 Monitor closely: attendance follow-up"
                if risk_level == "Medium"
                else "🟢 On track — routine check-in"
            ),
        }
    )


@app.route("/predict/score", methods=["POST"])
def predict_score():
    ensure_models_ready()
    data = get_json_body()
    data = enrich_features(data)

    row = pd.DataFrame([data]).reindex(columns=FEAT_SCORE, fill_value=0)
    pred_r = float(np.clip(model_score_r.predict(row)[0], 0, 100))
    pred_m = float(np.clip(model_score_m.predict(row)[0], 0, 100))
    avg = (pred_r + pred_m) / 2

    return jsonify(
        {
            "student_id": data.get("student_id", "unknown"),
            "predicted_score_reading": round(pred_r, 2),
            "predicted_score_math": round(pred_m, 2),
            "predicted_avg_score": round(avg, 2),
            "performance_band": (
                "Below Basic" if avg < 40 else "Basic" if avg < 55 else "Proficient" if avg < 70 else "Advanced"
            ),
        }
    )


@app.route("/cluster/assign", methods=["POST"])
def cluster_assign():
    ensure_models_ready()
    data = get_json_body()
    data = enrich_features(data)

    row = pd.DataFrame([data]).reindex(columns=FEAT_CLUSTER, fill_value=0)
    row_sc = scaler_cluster.transform(row)
    cluster_id = int(model_cluster.predict(row_sc)[0])
    risk_label = CLUSTER_RISK.get(cluster_id, "Unknown")

    return jsonify(
        {
            "student_id": data.get("student_id", "unknown"),
            "cluster_id": cluster_id,
            "risk_label": risk_label,
            "risk_icon": RISK_COLORS.get(risk_label, "⚪"),
        }
    )


@app.route("/simulate/policy", methods=["POST"])
def simulate_policy():
    ensure_models_ready()
    body = get_json_body()
    policies = body.pop("policies", [])
    if not isinstance(policies, list):
        return api_error("'policies' must be a list", 400)

    data = enrich_features(body)
    row_before = pd.DataFrame([data]).reindex(columns=FEAT_DROPOUT, fill_value=0)
    prob_before = float(model_dropout.predict_proba(row_before)[0, 1])

    data_after = data.copy()
    for policy in policies:
        if policy == "scholarship":
            data_after["scholarship"] = 1
        elif policy == "internet":
            data_after.update({"internet_home": 1, "internet_subsidy": 1, "digital_access": 1})
        elif policy == "device":
            data_after.update({"device_access": 1, "device_subsidy": 1, "digital_access": 1})
        elif policy == "remedial":
            data_after["remedial_program"] = 1
        elif policy == "free_lunch":
            data_after["free_lunch"] = 1
    data_after = enrich_features(data_after)

    row_after = pd.DataFrame([data_after]).reindex(columns=FEAT_DROPOUT, fill_value=0)
    prob_after = float(model_dropout.predict_proba(row_after)[0, 1])

    row_score_b = pd.DataFrame([data]).reindex(columns=FEAT_SCORE, fill_value=0)
    row_score_a = pd.DataFrame([data_after]).reindex(columns=FEAT_SCORE, fill_value=0)
    score_b = float(model_score_r.predict(row_score_b)[0])
    score_a = float(model_score_r.predict(row_score_a)[0])

    return jsonify(
        {
            "policies_applied": policies,
            "dropout_risk_before": round(prob_before, 4),
            "dropout_risk_after": round(prob_after, 4),
            "dropout_risk_change": round(prob_after - prob_before, 4),
            "reading_score_before": round(score_b, 2),
            "reading_score_after": round(score_a, 2),
            "reading_score_change": round(score_a - score_b, 2),
        }
    )


@app.route("/health", methods=["GET"])
def health():
    if MODEL_LOAD_ERROR is not None:
        return jsonify({"status": "error", "detail": MODEL_LOAD_ERROR}), 500
    return jsonify(
        {
            "status": "ok",
            "models": ["dropout", "score_reading", "score_math", "cluster"],
            "model1_auc": META["model1"].get("auc"),
            "model2_mae_r": META["model2"].get("mae_reading"),
            "model3_k": META["model3"].get("k"),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
