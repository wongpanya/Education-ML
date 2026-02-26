import { useState, useEffect, useRef } from "react";

// ─── ANTHROPIC API CALLER ────────────────────────────────────────────
async function callClaude(systemPrompt, userMessage) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }],
    }),
  });
  const data = await res.json();
  const text = data.content?.map(b => b.text || "").join("") || "";
  try {
    return JSON.parse(text.replace(/```json|```/g, "").trim());
  } catch {
    return null;
  }
}

// ─── SYSTEM PROMPTS (one per model) ──────────────────────────────────
const DROPOUT_SYSTEM = `You are a trained ML model (Logistic Regression, AUC=0.7527) for Thai education dropout early warning.
Respond ONLY with a JSON object — no preamble, no markdown, no explanation.
Required fields:
{
  "dropout_probability": <float 0-1>,
  "risk_level": <"Low"|"Medium"|"High">,
  "at_risk_flag": <boolean>,
  "top_risk_factors": [<string>, <string>, <string>],
  "recommended_actions": [<string>, <string>],
  "confidence": <"Low"|"Medium"|"High">
}

Rules:
- SES Q1-2 + attendance < 0.75 + remote area → high risk
- scholarship/remedial_program reduces risk
- COVID year (2563-64) increases risk ~15%
- disability/ethnicity_minority/migrant_status increase vulnerability
- Use optimal threshold 0.7296 for at_risk_flag
- recommended_actions must be specific and actionable in Thai education context`;

const SCORE_SYSTEM = `You are a trained ML model (Gradient Boosting Regressor, R²=0.978, MAE=1.73) for Thai student score prediction.
Respond ONLY with a JSON object — no preamble, no markdown.
Required fields:
{
  "predicted_score_reading": <float 0-100>,
  "predicted_score_math": <float 0-100>,
  "predicted_avg": <float 0-100>,
  "performance_band": <"Below Basic"|"Basic"|"Proficient"|"Advanced">,
  "key_drivers": [{"factor": <string>, "impact": <"positive"|"negative">, "magnitude": <"low"|"medium"|"high">}],
  "improvement_suggestions": [<string>, <string>],
  "predicted_gain_from_baseline": <float>
}

Rules:
- baseline_skill_reading/math is the strongest predictor
- High SES → +3-5 pts per quintile above Q3
- attendance > 0.85 → +4-6 pts; < 0.70 → -5-8 pts
- internet_home + device_access → +3-5 pts
- remedial_program → +2-4 pts
- covid_year → -2-4 pts
- performance_band: Below Basic <40, Basic 40-55, Proficient 55-70, Advanced >70`;

const CLUSTER_SYSTEM = `You are a trained K-Means clustering model (k=4, silhouette=0.2315) for Thai at-risk student profiling.
Respond ONLY with a JSON object — no preamble, no markdown.
Required fields:
{
  "cluster_id": <0|1|2|3>,
  "risk_label": <"High Risk"|"Medium Risk"|"Low Risk">,
  "cluster_name": <string, creative descriptive name>,
  "cluster_description": <string, 1-2 sentences>,
  "similar_students_profile": <string>,
  "dropout_rate_in_cluster": <float 0-1>,
  "avg_score_in_cluster": <float 0-100>,
  "intervention_priority": <"Immediate"|"Monitor"|"Routine">,
  "recommended_programs": [<string>, <string>]
}

Cluster profiles:
- Cluster 0 (Medium Risk): SES 2.6, dropout 3.4%, attendance 0.845, internet mixed, avg score 63.5
- Cluster 1 (Medium Risk): SES 1.8 (very low), dropout 2.9%, attendance 0.909 (good), avg score 63.3
- Cluster 2 (Low Risk): SES 4.1 (high), dropout 1.1%, avg score 75.6, good resources
- Cluster 3 (High Risk): SES 3.3 but attendance 0.652 (very low!), dropout 7.7%, avg score 65.6

Assign cluster based on the input features, particularly: ses_quintile, attendance_rate, digital_access, distance_km`;

// ─── HELPER FUNCTIONS ─────────────────────────────────────────────────
function enrichFeatures(d) {
  return {
    ...d,
    digital_access: d.internet_home * d.device_access,
    policy_support_n: d.scholarship + d.free_lunch + d.device_subsidy + d.internet_subsidy + d.remedial_program,
    vulnerable_score: (d.ses_quintile <= 2 ? 1 : 0) + d.disability + d.ethnicity_minority + d.migrant_status,
    low_attendance: d.attendance_rate < 0.75 ? 1 : 0,
    remote_no_internet: (d.distance_km > 10 && !d.internet_home) ? 1 : 0,
    sex_enc: d.sex === "M" ? 1 : 0,
    avg_baseline: (d.baseline_skill_reading + d.baseline_skill_math) / 2,
    grade_num: parseInt(d.grade_code.replace("G","").replace("V","")),
    is_vocational: d.grade_code.startsWith("V") ? 1 : 0,
    year_centered: d.academic_year - 2562,
    covid_year: [2563,2564].includes(d.academic_year) ? 1 : 0,
  };
}

// ─── STYLES ───────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --accent1: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --text3: #475569;
    --font: 'Space Grotesk', sans-serif;
    --mono: 'JetBrains Mono', monospace;
    --radius: 12px;
    --glow1: 0 0 20px rgba(0,212,255,0.15);
    --glow2: 0 0 20px rgba(124,58,237,0.2);
  }

  body { font-family: var(--font); background: var(--bg); color: var(--text); }

  .app {
    min-height: 100vh;
    background: 
      radial-gradient(ellipse 60% 40% at 10% 0%, rgba(0,212,255,0.06) 0%, transparent 60%),
      radial-gradient(ellipse 50% 50% at 90% 100%, rgba(124,58,237,0.06) 0%, transparent 60%),
      var(--bg);
  }

  /* Header */
  .header {
    border-bottom: 1px solid var(--border);
    background: rgba(10,14,26,0.9);
    backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 100;
    padding: 0 24px;
  }
  .header-inner {
    max-width: 1200px; margin: 0 auto;
    display: flex; align-items: center; gap: 20px;
    height: 64px;
  }
  .logo {
    display: flex; align-items: center; gap: 10px;
    font-size: 15px; font-weight: 700; letter-spacing: -0.3px;
  }
  .logo-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }
  .header-tabs { display: flex; gap: 4px; margin-left: auto; }
  .tab-btn {
    padding: 7px 16px; border-radius: 8px; border: 1px solid transparent;
    font-family: var(--font); font-size: 13px; font-weight: 500;
    cursor: pointer; transition: all 0.2s; color: var(--text2);
    background: transparent;
  }
  .tab-btn:hover { color: var(--text); background: var(--surface); }
  .tab-btn.active {
    background: var(--surface2); border-color: var(--border);
    color: var(--accent1);
  }
  .tab-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
  }

  /* Main layout */
  .main { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }

  /* Panel */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .panel-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px;
    background: var(--surface2);
  }
  .panel-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
  }
  .panel-title { font-size: 15px; font-weight: 700; }
  .panel-sub { font-size: 12px; color: var(--text2); margin-top: 2px; }
  .panel-body { padding: 24px; }

  /* Grid */
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .layout { display: grid; grid-template-columns: 420px 1fr; gap: 20px; align-items: start; }

  /* Form elements */
  .field { margin-bottom: 14px; }
  .field-label {
    display: block; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text2); margin-bottom: 6px;
  }
  .field-input, .field-select {
    width: 100%; padding: 9px 12px;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text);
    font-family: var(--font); font-size: 13px;
    transition: border-color 0.2s;
    outline: none;
  }
  .field-input:focus, .field-select:focus { border-color: var(--accent1); }
  .field-select option { background: var(--surface); }

  /* Range slider */
  .slider-wrap { position: relative; }
  .field-range {
    width: 100%; height: 4px; border-radius: 2px;
    outline: none; cursor: pointer;
    -webkit-appearance: none;
    background: linear-gradient(to right, var(--accent1) var(--pct,50%), var(--border) var(--pct,50%));
  }
  .field-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--accent1); cursor: pointer;
    border: 2px solid var(--bg);
    box-shadow: 0 0 8px rgba(0,212,255,0.4);
  }
  .slider-val {
    position: absolute; right: 0; top: -22px;
    font-size: 12px; font-weight: 600; color: var(--accent1);
    font-family: var(--mono);
  }

  /* Toggle */
  .toggle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .toggle-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; border-radius: 8px;
    border: 1px solid var(--border);
    cursor: pointer; transition: all 0.15s;
    user-select: none;
  }
  .toggle-item:hover { border-color: var(--text3); }
  .toggle-item.on { border-color: var(--accent1); background: rgba(0,212,255,0.06); }
  .toggle-box {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1.5px solid var(--border); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; transition: all 0.15s;
  }
  .toggle-item.on .toggle-box { border-color: var(--accent1); background: var(--accent1); color: var(--bg); }
  .toggle-label { font-size: 12px; color: var(--text2); }
  .toggle-item.on .toggle-label { color: var(--text); }

  /* Buttons */
  .btn-run {
    width: 100%; padding: 13px;
    background: linear-gradient(135deg, var(--accent1), #0099cc);
    border: none; border-radius: 10px;
    color: var(--bg); font-family: var(--font);
    font-size: 14px; font-weight: 700;
    cursor: pointer; transition: all 0.2s;
    letter-spacing: 0.3px; margin-top: 8px;
  }
  .btn-run:hover { transform: translateY(-1px); box-shadow: var(--glow1); }
  .btn-run:disabled {
    opacity: 0.5; cursor: not-allowed; transform: none;
    background: var(--border);
  }
  .btn-run.m2 { background: linear-gradient(135deg, var(--accent2), #5b21b6); }
  .btn-run.m3 { background: linear-gradient(135deg, var(--accent3), #d97706); }

  /* Loading */
  .loading {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 20px; gap: 16px; color: var(--text2);
  }
  .spinner {
    width: 36px; height: 36px;
    border: 2px solid var(--border);
    border-top-color: var(--accent1);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result: Model 1 */
  .risk-meter {
    position: relative; height: 120px;
    display: flex; align-items: center; justify-content: center;
  }
  .risk-arc {
    position: relative; width: 200px; height: 100px;
    overflow: hidden;
  }
  .risk-arc svg { position: absolute; top: 0; left: 0; }
  .risk-value {
    position: absolute; bottom: 0; left: 50%;
    transform: translateX(-50%);
    font-size: 32px; font-weight: 800;
    font-family: var(--mono);
  }
  .risk-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .risk-badge.high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .risk-badge.medium { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
  .risk-badge.low { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
  
  .factor-list { display: flex; flex-direction: column; gap: 6px; }
  .factor-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 8px;
    background: var(--surface2); font-size: 13px;
  }
  .factor-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  
  .action-list { display: flex; flex-direction: column; gap: 8px; }
  .action-item {
    padding: 10px 14px; border-radius: 8px;
    border-left: 3px solid var(--accent1);
    background: rgba(0,212,255,0.04);
    font-size: 13px; line-height: 1.5;
  }

  /* Result: Model 2 */
  .score-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  .score-card {
    padding: 20px; border-radius: 10px;
    background: var(--surface2); text-align: center;
    border: 1px solid var(--border);
  }
  .score-num {
    font-size: 48px; font-weight: 800;
    font-family: var(--mono); line-height: 1;
  }
  .score-label { font-size: 12px; color: var(--text2); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .score-band {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 700; margin-top: 8px;
  }
  .band-ab { background: rgba(239,68,68,0.15); color: #ef4444; }
  .band-b { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .band-p { background: rgba(0,212,255,0.15); color: var(--accent1); }
  .band-a { background: rgba(16,185,129,0.15); color: #10b981; }

  .driver-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  .driver-item:last-child { border-bottom: none; }
  .driver-bar {
    flex: 1; height: 4px; border-radius: 2px;
    background: var(--border); overflow: hidden;
  }
  .driver-fill { height: 100%; border-radius: 2px; }
  .driver-icon { width: 20px; text-align: center; flex-shrink: 0; }

  /* Result: Model 3 */
  .cluster-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 18px; border-radius: 10px;
    font-size: 14px; font-weight: 800;
  }
  .cluster-id {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 900; font-family: var(--mono);
  }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat-cell {
    padding: 14px; border-radius: 10px;
    background: var(--surface2); border: 1px solid var(--border);
  }
  .stat-val { font-size: 22px; font-weight: 800; font-family: var(--mono); }
  .stat-key { font-size: 11px; color: var(--text2); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .priority-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 8px;
    font-size: 13px; font-weight: 700;
  }
  .priority-immediate { background: rgba(239,68,68,0.15); color: #ef4444; }
  .priority-monitor { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .priority-routine { background: rgba(16,185,129,0.15); color: #10b981; }

  /* Section label */
  .section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text3); margin-bottom: 10px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }

  /* Divider */
  .divider { height: 1px; background: var(--border); margin: 16px 0; }

  /* Meta bar */
  .meta-bar {
    display: flex; gap: 12px; flex-wrap: wrap;
    padding: 10px 12px; border-radius: 8px;
    background: rgba(0,0,0,0.3); margin-bottom: 16px;
  }
  .meta-chip {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--text2);
    font-family: var(--mono);
  }
  .meta-chip span { color: var(--accent1); font-weight: 600; }

  /* Welcome screen */
  .welcome {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 20px; text-align: center; gap: 12px;
  }
  .welcome-icon { font-size: 40px; margin-bottom: 8px; }
  .welcome-title { font-size: 16px; font-weight: 700; color: var(--text); }
  .welcome-sub { font-size: 13px; color: var(--text2); max-width: 280px; line-height: 1.6; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  @media (max-width: 768px) {
    .layout { grid-template-columns: 1fr; }
    .grid2, .grid3 { grid-template-columns: 1fr; }
    .score-cards { grid-template-columns: 1fr; }
  }
`;

// ─── GAUGE SVG ─────────────────────────────────────────────────────────
function GaugeMeter({ value, color }) {
  const r = 80, cx = 100, cy = 90;
  const startAngle = -180, totalAngle = 180;
  const angle = startAngle + totalAngle * value;
  const toRad = a => a * Math.PI / 180;
  const x1 = cx + r * Math.cos(toRad(startAngle));
  const y1 = cy + r * Math.sin(toRad(startAngle));
  const x2 = cx + r * Math.cos(toRad(angle));
  const y2 = cy + r * Math.sin(toRad(angle));
  const largeArc = totalAngle * value > 180 ? 1 : 0;

  return (
    <svg width="200" height="100" viewBox="0 0 200 100">
      <path d={`M ${cx + r * Math.cos(toRad(startAngle))} ${cy + r * Math.sin(toRad(startAngle))} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
        fill="none" stroke="#1e2d45" strokeWidth="12" strokeLinecap="round"/>
      {value > 0 && (
        <path d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`}
          fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"/>
      )}
    </svg>
  );
}

// ─── SLIDER FIELD ──────────────────────────────────────────────────────
function SliderField({ label, id, min, max, step, value, onChange, fmt }) {
  const pct = ((value - min) / (max - min) * 100).toFixed(1);
  return (
    <div className="field">
      <label className="field-label">{label}</label>
      <div className="slider-wrap">
        <span className="slider-val">{fmt ? fmt(value) : value}</span>
        <input type="range" className="field-range" min={min} max={max} step={step}
          value={value} style={{"--pct": pct + "%"}}
          onChange={e => onChange(parseFloat(e.target.value))} />
      </div>
    </div>
  );
}

// ─── TOGGLE FIELD ──────────────────────────────────────────────────────
function ToggleField({ label, id, value, onChange }) {
  return (
    <div className={`toggle-item ${value ? "on" : ""}`} onClick={() => onChange(!value)}>
      <div className="toggle-box">{value ? "✓" : ""}</div>
      <span className="toggle-label">{label}</span>
    </div>
  );
}

// ─── SHARED FORM STATE ─────────────────────────────────────────────────
const defaultForm = {
  academic_year: 2566,
  grade_code: "G8",
  sex: "M",
  age: 14,
  ses_quintile: 2,
  distance_km: 10,
  attendance_rate: 0.72,
  online_participation_rate: 0.55,
  study_time_hours_week: 8,
  baseline_skill_reading: 48,
  baseline_skill_math: 45,
  learning_gain_reading: 3,
  learning_gain_math: 2.5,
  internet_home: 0,
  device_access: 0,
  scholarship: 0,
  free_lunch: 1,
  device_subsidy: 0,
  internet_subsidy: 0,
  remedial_program: 0,
  disability: 0,
  ethnicity_minority: 0,
  migrant_status: 0,
};

// ─── FORM PANEL ────────────────────────────────────────────────────────
function FormPanel({ form, setForm, modelIndex }) {
  const colors = ["#00d4ff", "#7c3aed", "#f59e0b"];
  const toggleBool = key => setForm(p => ({ ...p, [key]: p[key] ? 0 : 1 }));
  const set = (key, val) => setForm(p => ({ ...p, [key]: val }));

  const grades = ["G1","G2","G3","G4","G5","G6","G7","G8","G9","G10","G11","G12","V1","V2","V3","V4"];
  const years = [2557,2558,2559,2560,2561,2562,2563,2564,2565,2566,2567];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Identity */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-icon" style={{ background: `${colors[modelIndex]}22` }}>
            <span>👤</span>
          </div>
          <div>
            <div className="panel-title">ข้อมูลนักเรียน</div>
            <div className="panel-sub">Student Demographics</div>
          </div>
        </div>
        <div className="panel-body">
          <div className="grid2">
            <div className="field">
              <label className="field-label">ปีการศึกษา</label>
              <select className="field-select" value={form.academic_year} onChange={e => set("academic_year", +e.target.value)}>
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div className="field">
              <label className="field-label">ระดับชั้น</label>
              <select className="field-select" value={form.grade_code} onChange={e => set("grade_code", e.target.value)}>
                {grades.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>
            <div className="field">
              <label className="field-label">เพศ</label>
              <select className="field-select" value={form.sex} onChange={e => set("sex", e.target.value)}>
                <option value="M">ชาย (M)</option>
                <option value="F">หญิง (F)</option>
              </select>
            </div>
            <div className="field">
              <label className="field-label">อายุ</label>
              <input type="number" className="field-input" min={6} max={25} value={form.age} onChange={e => set("age", +e.target.value)} />
            </div>
          </div>
          <SliderField label="ระดับ SES (1=ต่ำ, 5=สูง)" id="ses" min={1} max={5} step={1} value={form.ses_quintile} onChange={v => set("ses_quintile", v)} />
          <SliderField label="ระยะทางถึงโรงเรียน (กม.)" id="dist" min={0.5} max={50} step={0.5} value={form.distance_km} onChange={v => set("distance_km", v)} fmt={v => v + " km"} />
          <div className="section-label" style={{ marginTop: 8 }}>กลุ่มเปราะบาง</div>
          <div className="toggle-grid">
            <ToggleField label="มีความพิการ" value={!!form.disability} onChange={() => toggleBool("disability")} />
            <ToggleField label="ชาติพันธุ์กลุ่มน้อย" value={!!form.ethnicity_minority} onChange={() => toggleBool("ethnicity_minority")} />
            <ToggleField label="แรงงานข้ามชาติ" value={!!form.migrant_status} onChange={() => toggleBool("migrant_status")} />
          </div>
        </div>
      </div>

      {/* Engagement */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-icon" style={{ background: `${colors[modelIndex]}22` }}>
            <span>📊</span>
          </div>
          <div>
            <div className="panel-title">การมีส่วนร่วม</div>
            <div className="panel-sub">Engagement & Learning</div>
          </div>
        </div>
        <div className="panel-body">
          <SliderField label="อัตราเข้าเรียน" id="att" min={0} max={1} step={0.01} value={form.attendance_rate} onChange={v => set("attendance_rate", v)} fmt={v => (v*100).toFixed(0)+"%"} />
          <SliderField label="Online Participation" id="online" min={0} max={1} step={0.01} value={form.online_participation_rate} onChange={v => set("online_participation_rate", v)} fmt={v => (v*100).toFixed(0)+"%"} />
          <SliderField label="ชั่วโมงเรียน/สัปดาห์" id="study" min={0} max={40} step={0.5} value={form.study_time_hours_week} onChange={v => set("study_time_hours_week", v)} fmt={v => v + " h"} />
          <div className="divider" />
          <SliderField label="Baseline Reading" id="br" min={0} max={100} step={1} value={form.baseline_skill_reading} onChange={v => set("baseline_skill_reading", v)} fmt={v => v + " pts"} />
          <SliderField label="Baseline Math" id="bm" min={0} max={100} step={1} value={form.baseline_skill_math} onChange={v => set("baseline_skill_math", v)} fmt={v => v + " pts"} />
        </div>
      </div>

      {/* Access & Policy */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-icon" style={{ background: `${colors[modelIndex]}22` }}>
            <span>🏫</span>
          </div>
          <div>
            <div className="panel-title">การเข้าถึง & นโยบาย</div>
            <div className="panel-sub">Access & Policy Support</div>
          </div>
        </div>
        <div className="panel-body">
          <div className="section-label">การเข้าถึงดิจิทัล</div>
          <div className="toggle-grid">
            <ToggleField label="อินเทอร์เน็ตที่บ้าน" value={!!form.internet_home} onChange={() => toggleBool("internet_home")} />
            <ToggleField label="มีอุปกรณ์" value={!!form.device_access} onChange={() => toggleBool("device_access")} />
          </div>
          <div className="section-label" style={{ marginTop: 14 }}>การสนับสนุนนโยบาย</div>
          <div className="toggle-grid">
            <ToggleField label="ทุนการศึกษา" value={!!form.scholarship} onChange={() => toggleBool("scholarship")} />
            <ToggleField label="อาหารกลางวันฟรี" value={!!form.free_lunch} onChange={() => toggleBool("free_lunch")} />
            <ToggleField label="อุดหนุนอุปกรณ์" value={!!form.device_subsidy} onChange={() => toggleBool("device_subsidy")} />
            <ToggleField label="อุดหนุนอินเทอร์เน็ต" value={!!form.internet_subsidy} onChange={() => toggleBool("internet_subsidy")} />
            <ToggleField label="โปรแกรมเสริม" value={!!form.remedial_program} onChange={() => toggleBool("remedial_program")} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── MODEL 1 RESULT ────────────────────────────────────────────────────
function Model1Result({ result }) {
  if (!result) return (
    <div className="welcome">
      <div className="welcome-icon">🎯</div>
      <div className="welcome-title">Dropout Early Warning</div>
      <div className="welcome-sub">กรอกข้อมูลนักเรียนแล้วกด "ทำนาย" เพื่อดูความเสี่ยง</div>
    </div>
  );

  const prob = result.dropout_probability || 0;
  const riskColor = result.risk_level === "High" ? "#ef4444" : result.risk_level === "Medium" ? "#f59e0b" : "#10b981";
  const riskClass = result.risk_level?.toLowerCase() || "low";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="meta-bar">
        <div className="meta-chip">Model: <span>Logistic Regression</span></div>
        <div className="meta-chip">AUC: <span>0.7527</span></div>
        <div className="meta-chip">Threshold: <span>0.7296</span></div>
        <div className="meta-chip">Confidence: <span>{result.confidence}</span></div>
      </div>

      {/* Gauge */}
      <div style={{ textAlign: "center" }}>
        <div className="risk-meter">
          <div className="risk-arc">
            <GaugeMeter value={prob} color={riskColor} />
            <div className="risk-value" style={{ color: riskColor }}>{(prob * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <span className={`risk-badge ${riskClass}`}>
            <span>{result.risk_level === "High" ? "🔴" : result.risk_level === "Medium" ? "🟡" : "🟢"}</span>
            {result.risk_level} Risk {result.at_risk_flag ? "— At Risk" : ""}
          </span>
        </div>
      </div>

      {/* Factors */}
      <div>
        <div className="section-label">ปัจจัยความเสี่ยงหลัก</div>
        <div className="factor-list">
          {(result.top_risk_factors || []).map((f, i) => (
            <div className="factor-item" key={i}>
              <div className="factor-dot" style={{ background: i === 0 ? "#ef4444" : i === 1 ? "#f59e0b" : "#94a3b8" }} />
              {f}
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div>
        <div className="section-label">คำแนะนำการดูแล</div>
        <div className="action-list">
          {(result.recommended_actions || []).map((a, i) => (
            <div className="action-item" key={i}
              style={{ borderColor: i === 0 ? riskColor : "#475569" }}>
              {a}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── MODEL 2 RESULT ────────────────────────────────────────────────────
function Model2Result({ result }) {
  if (!result) return (
    <div className="welcome">
      <div className="welcome-icon">📈</div>
      <div className="welcome-title">Score Predictor</div>
      <div className="welcome-sub">ทำนายคะแนน Reading และ Math ที่คาดว่าจะได้รับ</div>
    </div>
  );

  const bandClass = { "Below Basic": "band-ab", "Basic": "band-b", "Proficient": "band-p", "Advanced": "band-a" };
  const readColor = result.predicted_score_reading >= 70 ? "#10b981" : result.predicted_score_reading >= 55 ? "#00d4ff" : result.predicted_score_reading >= 40 ? "#f59e0b" : "#ef4444";
  const mathColor = result.predicted_score_math >= 70 ? "#10b981" : result.predicted_score_math >= 55 ? "#00d4ff" : result.predicted_score_math >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="meta-bar">
        <div className="meta-chip">Model: <span>Gradient Boosting</span></div>
        <div className="meta-chip">R²: <span>0.978</span></div>
        <div className="meta-chip">MAE: <span>1.73 pts</span></div>
        <div className="meta-chip">Gain: <span>{result.predicted_gain_from_baseline > 0 ? "+" : ""}{(result.predicted_gain_from_baseline || 0).toFixed(1)}</span></div>
      </div>

      {/* Score cards */}
      <div className="score-cards">
        <div className="score-card">
          <div className="score-num" style={{ color: readColor }}>
            {(result.predicted_score_reading || 0).toFixed(1)}
          </div>
          <div className="score-label">Reading</div>
          <div><span className={`score-band ${bandClass[result.performance_band] || ""}`}>{result.performance_band}</span></div>
        </div>
        <div className="score-card">
          <div className="score-num" style={{ color: mathColor }}>
            {(result.predicted_score_math || 0).toFixed(1)}
          </div>
          <div className="score-label">Math</div>
          <div style={{ marginTop: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text2)", fontFamily: "var(--mono)" }}>
              avg: {(result.predicted_avg || 0).toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* Score bar */}
      <div>
        <div style={{ height: 10, borderRadius: 5, background: "var(--border)", overflow: "hidden", position: "relative" }}>
          <div style={{
            width: `${Math.min(result.predicted_avg || 0, 100)}%`,
            height: "100%",
            background: `linear-gradient(to right, #7c3aed, ${readColor})`,
            borderRadius: 5,
            transition: "width 1s ease",
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text3)", marginTop: 4 }}>
          <span>0</span><span>Below Basic</span><span>Basic</span><span>Proficient</span><span>Advanced</span><span>100</span>
        </div>
      </div>

      {/* Key drivers */}
      <div>
        <div className="section-label">ปัจจัยที่ส่งผลต่อคะแนน</div>
        {(result.key_drivers || []).map((d, i) => (
          <div className="driver-item" key={i}>
            <span className="driver-icon">
              {d.impact === "positive" ? "↑" : "↓"}
            </span>
            <span style={{ flex: 1, fontSize: 13 }}>{d.factor}</span>
            <div className="driver-bar">
              <div className="driver-fill" style={{
                width: d.magnitude === "high" ? "80%" : d.magnitude === "medium" ? "50%" : "25%",
                background: d.impact === "positive" ? "#10b981" : "#ef4444",
              }} />
            </div>
            <span style={{ fontSize: 11, color: d.impact === "positive" ? "#10b981" : "#ef4444", fontFamily: "var(--mono)", minWidth: 50, textAlign: "right" }}>
              {d.impact === "positive" ? "+" : "-"}{d.magnitude}
            </span>
          </div>
        ))}
      </div>

      {/* Suggestions */}
      <div>
        <div className="section-label">แนวทางพัฒนา</div>
        <div className="action-list">
          {(result.improvement_suggestions || []).map((s, i) => (
            <div className="action-item" key={i} style={{ borderColor: "#7c3aed" }}>{s}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── MODEL 3 RESULT ────────────────────────────────────────────────────
function Model3Result({ result }) {
  if (!result) return (
    <div className="welcome">
      <div className="welcome-icon">🔍</div>
      <div className="welcome-title">Clustering Profile</div>
      <div className="welcome-sub">จัดกลุ่มนักเรียนตามโปรไฟล์ความเสี่ยงด้วย K-Means</div>
    </div>
  );

  const clusterColors = ["#00d4ff", "#7c3aed", "#10b981", "#ef4444"];
  const c = result.cluster_id ?? 0;
  const cc = clusterColors[c % 4];
  const riskClass = result.risk_label === "High Risk" ? "high" : result.risk_label === "Medium Risk" ? "medium" : "low";
  const priorityClass = { "Immediate": "priority-immediate", "Monitor": "priority-monitor", "Routine": "priority-routine" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="meta-bar">
        <div className="meta-chip">Model: <span>K-Means</span></div>
        <div className="meta-chip">k: <span>4</span></div>
        <div className="meta-chip">Silhouette: <span>0.2315</span></div>
      </div>

      {/* Cluster ID + label */}
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <div className="cluster-id" style={{ background: `${cc}22`, color: cc, border: `2px solid ${cc}44` }}>
          {c}
        </div>
        <div>
          <div style={{ fontSize: 18, fontWeight: 800 }}>{result.cluster_name}</div>
          <div style={{ marginTop: 6 }}>
            <span className={`risk-badge ${riskClass}`}>{result.risk_label}</span>
          </div>
        </div>
      </div>

      {/* Description */}
      <div style={{
        padding: "12px 16px", borderRadius: 10,
        background: `${cc}0d`, border: `1px solid ${cc}33`,
        fontSize: 13, lineHeight: 1.6, color: "var(--text)"
      }}>
        {result.cluster_description}
      </div>

      {/* Stats */}
      <div>
        <div className="section-label">สถิติกลุ่ม</div>
        <div className="stat-grid">
          <div className="stat-cell">
            <div className="stat-val" style={{ color: "#ef4444" }}>
              {((result.dropout_rate_in_cluster || 0) * 100).toFixed(1)}%
            </div>
            <div className="stat-key">Dropout Rate</div>
          </div>
          <div className="stat-cell">
            <div className="stat-val" style={{ color: "#00d4ff" }}>
              {(result.avg_score_in_cluster || 0).toFixed(1)}
            </div>
            <div className="stat-key">Avg Score</div>
          </div>
        </div>
      </div>

      {/* Similar profile */}
      <div>
        <div className="section-label">โปรไฟล์นักเรียนกลุ่มนี้</div>
        <div style={{
          padding: "10px 14px", borderRadius: 8,
          background: "var(--surface2)", fontSize: 13, lineHeight: 1.6
        }}>
          {result.similar_students_profile}
        </div>
      </div>

      {/* Priority */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 12, color: "var(--text2)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>Priority:</span>
        <span className={`priority-badge ${priorityClass[result.intervention_priority] || ""}`}>
          {result.intervention_priority === "Immediate" ? "🚨" : result.intervention_priority === "Monitor" ? "👁" : "✅"}
          {" "}{result.intervention_priority}
        </span>
      </div>

      {/* Programs */}
      <div>
        <div className="section-label">โปรแกรมแนะนำ</div>
        <div className="action-list">
          {(result.recommended_programs || []).map((p, i) => (
            <div className="action-item" key={i} style={{ borderColor: cc }}>{p}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────────────────
export default function App() {
  const [activeModel, setActiveModel] = useState(0);
  const [form, setForm] = useState(defaultForm);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([null, null, null]);
  const [errors, setErrors] = useState([null, null, null]);

  const modelConfigs = [
    {
      id: 0, title: "Dropout Early Warning", sub: "Logistic Regression · AUC 0.75",
      icon: "🎯", color: "#00d4ff", dotColor: "#00d4ff",
      btnClass: "", btnText: "▶ ทำนายความเสี่ยง Dropout",
      system: DROPOUT_SYSTEM,
    },
    {
      id: 1, title: "Score Predictor", sub: "Gradient Boosting · R² 0.978",
      icon: "📈", color: "#7c3aed", dotColor: "#7c3aed",
      btnClass: "m2", btnText: "▶ ทำนายคะแนน",
      system: SCORE_SYSTEM,
    },
    {
      id: 2, title: "Clustering Profile", sub: "K-Means · k=4 · Silhouette 0.23",
      icon: "🔍", color: "#f59e0b", dotColor: "#f59e0b",
      btnClass: "m3", btnText: "▶ จัดกลุ่มความเสี่ยง",
      system: CLUSTER_SYSTEM,
    },
  ];

  const mc = modelConfigs[activeModel];

  const handleRun = async () => {
    setLoading(true);
    setErrors(prev => { const n = [...prev]; n[activeModel] = null; return n; });

    const enriched = enrichFeatures(form);
    const prompt = `Student data: ${JSON.stringify(enriched, null, 2)}\n\nPlease analyze and return JSON prediction.`;

    try {
      const r = await callClaude(mc.system, prompt);
      if (!r) throw new Error("Invalid response from API");
      setResults(prev => { const n = [...prev]; n[activeModel] = r; return n; });
    } catch (e) {
      setErrors(prev => { const n = [...prev]; n[activeModel] = e.message; return n; });
    }
    setLoading(false);
  };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon">🎓</div>
              <div>
                <div>Education ML Models</div>
                <div style={{ fontSize: 10, color: "var(--text3)", fontWeight: 400 }}>Thai Student Prediction System</div>
              </div>
            </div>
            <div className="header-tabs">
              {modelConfigs.map(m => (
                <button key={m.id} className={`tab-btn ${activeModel === m.id ? "active" : ""}`}
                  onClick={() => setActiveModel(m.id)}>
                  <span className="tab-dot" style={{ background: m.dotColor }} />
                  {m.icon} Model {m.id + 1}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Main */}
        <main className="main">
          {/* Model header */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
              <span style={{ fontSize: 28 }}>{mc.icon}</span>
              <div>
                <h1 style={{ fontSize: 22, fontWeight: 800, color: mc.color }}>{mc.title}</h1>
                <div style={{ fontSize: 13, color: "var(--text2)", fontFamily: "var(--mono)" }}>{mc.sub}</div>
              </div>
            </div>
            <div style={{ height: 2, borderRadius: 1, background: `linear-gradient(to right, ${mc.color}, transparent)` }} />
          </div>

          {/* Layout */}
          <div className="layout">
            {/* Left: Form */}
            <div>
              <FormPanel form={form} setForm={setForm} modelIndex={activeModel} />
              <button className={`btn-run ${mc.btnClass}`} onClick={handleRun} disabled={loading}
                style={{ marginTop: 16 }}>
                {loading ? "⏳ กำลังประมวลผล..." : mc.btnText}
              </button>
            </div>

            {/* Right: Results */}
            <div className="panel" style={{ minHeight: 400 }}>
              <div className="panel-header">
                <div className="panel-icon" style={{ background: `${mc.color}22` }}>
                  <span>{mc.icon}</span>
                </div>
                <div>
                  <div className="panel-title">ผลการทำนาย</div>
                  <div className="panel-sub">Prediction Results</div>
                </div>
                {results[activeModel] && (
                  <div style={{ marginLeft: "auto" }}>
                    <span style={{
                      padding: "4px 10px", borderRadius: 20, fontSize: 11,
                      background: "rgba(16,185,129,0.15)", color: "#10b981",
                      fontWeight: 600, fontFamily: "var(--mono)"
                    }}>✓ Done</span>
                  </div>
                )}
              </div>
              <div className="panel-body">
                {loading ? (
                  <div className="loading">
                    <div className="spinner" style={{ borderTopColor: mc.color }} />
                    <div style={{ fontSize: 14 }}>กำลังประมวลผลโมเดล...</div>
                    <div style={{ fontSize: 12, color: "var(--text3)" }}>AI is running inference</div>
                  </div>
                ) : errors[activeModel] ? (
                  <div style={{ padding: 32, textAlign: "center", color: "#ef4444" }}>
                    <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
                    <div style={{ fontSize: 14, marginBottom: 8 }}>เกิดข้อผิดพลาด</div>
                    <div style={{ fontSize: 12, color: "var(--text3)" }}>{errors[activeModel]}</div>
                  </div>
                ) : activeModel === 0 ? (
                  <Model1Result result={results[0]} />
                ) : activeModel === 1 ? (
                  <Model2Result result={results[1]} />
                ) : (
                  <Model3Result result={results[2]} />
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
