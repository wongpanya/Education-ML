"""
╔══════════════════════════════════════════════════════════════════════╗
║  ML PIPELINE: Education Outcome Prediction                          ║
║  จาก student_year_mechanism.csv → 3 Models → Deployment            ║
║                                                                      ║
║  Model 1: Dropout Early Warning      (Classification)               ║
║  Model 2: Score Prediction           (Regression)                   ║
║  Model 3: At-Risk Student Clustering (Unsupervised)                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, warnings, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Output dirs ──────────────────────────────────────────────────────
OUT_FIGS  = "/mnt/user-data/outputs/ml_pipeline/figures"
OUT_MODEL = "/mnt/user-data/outputs/ml_pipeline/models"
OUT_REPORT= "/mnt/user-data/outputs/ml_pipeline"
for d in [OUT_FIGS, OUT_MODEL, OUT_REPORT]:
    os.makedirs(d, exist_ok=True)

# ── Plotting style ───────────────────────────────────────────────────
PALETTE   = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
             "#44BBA4", "#E94F37", "#393E41"]
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

print("=" * 65)
print("  ML PIPELINE: Education Outcome Prediction")
print("=" * 65)

# ════════════════════════════════════════════════════════════════════
# 0.  LOAD & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
print("\n[0] Loading & Feature Engineering...")

df = pd.read_csv("/mnt/user-data/outputs/dataset_b/student_year_mechanism.csv")

# ── derived features ─────────────────────────────────────────────
df["avg_score"]         = (df["score_reading"] + df["score_math"]) / 2
df["avg_baseline"]      = (df["baseline_skill_reading"] + df["baseline_skill_math"]) / 2
df["avg_gain"]          = (df["learning_gain_reading"] + df["learning_gain_math"]) / 2
df["score_gap"]         = df["score_reading"] - df["score_math"]
df["low_attendance"]    = (df["attendance_rate"] < 0.75).astype(int)
df["digital_access"]    = df["internet_home"] * df["device_access"]
df["policy_support_n"]  = (df[["scholarship","free_lunch","device_subsidy",
                                "internet_subsidy","remedial_program"]].sum(axis=1))
df["vulnerable_score"]  = ((df["ses_quintile"] <= 2).astype(int)
                           + df["disability"] + df["ethnicity_minority"]
                           + df["migrant_status"])
df["grade_num"]         = df["grade_code"].str.extract(r"(\d+)").astype(int)
df["is_vocational"]     = df["grade_code"].str.startswith("V").astype(int)
df["year_centered"]     = df["academic_year"] - 2562
df["covid_year"]        = df["academic_year"].isin([2563, 2564]).astype(int)
df["remote_no_internet"]= ((df["distance_km"] > 10) & (df["internet_home"] == 0)).astype(int)

# encode sex
df["sex_enc"] = (df["sex"] == "M").astype(int)

FEATURES_BASE = [
    # demographics
    "sex_enc","age","disability","ethnicity_minority","migrant_status",
    "ses_quintile","grade_num","is_vocational",
    # access & context
    "distance_km","transport_cost","internet_home","device_access","digital_access",
    "remote_no_internet",
    # policy
    "scholarship","free_lunch","device_subsidy","internet_subsidy",
    "remedial_program","policy_support_n",
    # engagement
    "attendance_rate","online_participation_rate","study_time_hours_week","low_attendance",
    # skills & learning
    "avg_baseline","avg_gain","score_gap",
    # vulnerability
    "vulnerable_score",
    # time
    "year_centered","covid_year",
]

print(f"   Rows: {len(df):,} | Features: {len(FEATURES_BASE)} | "
      f"Dropout rate: {df['dropout'].mean():.2%}")


# ════════════════════════════════════════════════════════════════════
# 1.  MODEL 1 — DROPOUT EARLY WARNING (Classification)
# ════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("[1] MODEL 1: Dropout Early Warning Classifier")
print("─"*65)

# ── features: exclude scores (รู้หลังจาก dropout เกิด) ──────────
FEAT_DROPOUT = [f for f in FEATURES_BASE
                if f not in ["avg_score","score_gap","avg_gain",
                              "score_reading","score_math"]]
FEAT_DROPOUT += ["avg_baseline"]   # ทักษะต้นปีรู้ได้ก่อน

X1 = df[FEAT_DROPOUT].fillna(0)
y1 = df["dropout"]

X1_tr, X1_te, y1_tr, y1_te = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1)

# sample weights สำหรับ imbalance (dropout=1 หายาก ~2.3%)
sw_tr = compute_sample_weight("balanced", y1_tr)

models_m1 = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, C=0.5))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
}

results_m1 = {}
for name, mdl in models_m1.items():
    if name == "Gradient Boosting":
        mdl.fit(X1_tr, y1_tr, sample_weight=sw_tr)
    else:
        mdl.fit(X1_tr, y1_tr)

    proba = mdl.predict_proba(X1_te)[:, 1]
    pred  = mdl.predict(X1_te)
    results_m1[name] = {
        "model"  : mdl,
        "proba"  : proba,
        "pred"   : pred,
        "auc"    : roc_auc_score(y1_te, proba),
        "ap"     : average_precision_score(y1_te, proba),
    }
    print(f"   {name:<25} AUC={results_m1[name]['auc']:.4f}  "
          f"AP={results_m1[name]['ap']:.4f}")

# ── choose best ──────────────────────────────────────────────────
best_m1_name = max(results_m1, key=lambda n: results_m1[n]["auc"])
best_m1      = results_m1[best_m1_name]["model"]
best_m1_prob = results_m1[best_m1_name]["proba"]
print(f"\n   ✓ Best: {best_m1_name} (AUC={results_m1[best_m1_name]['auc']:.4f})")

# ── feature importance ──────────────────────────────────────────
if hasattr(best_m1, "feature_importances_"):
    fi1 = pd.Series(best_m1.feature_importances_, index=FEAT_DROPOUT).sort_values(ascending=False)
else:
    fi_res = permutation_importance(best_m1, X1_te, y1_te, n_repeats=10, random_state=42)
    fi1 = pd.Series(fi_res.importances_mean, index=FEAT_DROPOUT).sort_values(ascending=False)

# ── optimal threshold (F2, เน้น recall) ─────────────────────────
prec, rec, thresholds = precision_recall_curve(y1_te, best_m1_prob)
f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
best_thr = thresholds[np.argmax(f2[:-1])]
y1_pred_opt = (best_m1_prob >= best_thr).astype(int)

print(f"\n   Optimal threshold (F2): {best_thr:.3f}")
print(classification_report(y1_te, y1_pred_opt,
                              target_names=["Not Dropout","Dropout"],
                              digits=3))


# ════════════════════════════════════════════════════════════════════
# 2.  MODEL 2 — SCORE PREDICTION (Regression)
# ════════════════════════════════════════════════════════════════════
print("─"*65)
print("[2] MODEL 2: Academic Score Predictor (Reading + Math)")
print("─"*65)

FEAT_SCORE = [f for f in FEATURES_BASE
              if f not in ["avg_score","score_reading","score_math"]]

df_enr = df[df["dropout"] == 0].copy()   # เฉพาะนักเรียนที่ไม่ dropout

for target, label in [("score_reading","Reading"), ("score_math","Math")]:
    X2 = df_enr[FEAT_SCORE].fillna(0)
    y2 = df_enr[target]

    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42)

    mdl_r = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42))
    ])
    mdl_r.fit(X2_tr, y2_tr)
    y2_pred = mdl_r.predict(X2_te)

    mae  = mean_absolute_error(y2_te, y2_pred)
    rmse = np.sqrt(mean_squared_error(y2_te, y2_pred))
    r2   = r2_score(y2_te, y2_pred)

    print(f"   {label:<10}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    if target == "score_reading":
        best_m2_r = mdl_r; X2_te_r = X2_te; y2_te_r = y2_te; pred_r = y2_pred
    else:
        best_m2_m = mdl_r; X2_te_m = X2_te; y2_te_m = y2_te; pred_m = y2_pred

# feature importance for score_reading
fi2 = pd.Series(
    mdl_r.named_steps["reg"].feature_importances_,
    index=FEAT_SCORE
).sort_values(ascending=False)


# ════════════════════════════════════════════════════════════════════
# 3.  MODEL 3 — AT-RISK CLUSTERING (Unsupervised)
# ════════════════════════════════════════════════════════════════════
print("─"*65)
print("[3] MODEL 3: At-Risk Student Profiling (K-Means Clustering)")
print("─"*65)

FEAT_CLUSTER = [
    "ses_quintile","avg_baseline","attendance_rate",
    "online_participation_rate","distance_km","digital_access",
    "vulnerable_score","policy_support_n","study_time_hours_week",
    "low_attendance","remote_no_internet","covid_year",
]

X3 = df[FEAT_CLUSTER].fillna(0)
scaler3 = StandardScaler()
X3_sc   = scaler3.fit_transform(X3)

# Elbow + Silhouette
sil_scores, inertias = [], []
K_RANGE = range(3, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(X3_sc)
    sil_scores.append(silhouette_score(X3_sc, labs))
    inertias.append(km.inertia_)

best_k = list(K_RANGE)[np.argmax(sil_scores)]
print(f"   Best K={best_k}  (silhouette={max(sil_scores):.4f})")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_best.fit_predict(X3_sc)

cluster_profile = df.groupby("cluster").agg(
    n=("student_id","count"),
    dropout_rate=("dropout","mean"),
    avg_score=("avg_score","mean"),
    avg_ses=("ses_quintile","mean"),
    avg_attendance=("attendance_rate","mean"),
    avg_internet=("internet_home","mean"),
    avg_distance=("distance_km","mean"),
    avg_policy_support=("policy_support_n","mean"),
).round(3)
cluster_profile["risk_label"] = ""
dr = cluster_profile["dropout_rate"]
cluster_profile["risk_label"] = pd.cut(
    dr, bins=[-1, dr.quantile(0.33), dr.quantile(0.67), 1],
    labels=["Low Risk","Medium Risk","High Risk"])
print(cluster_profile[["n","dropout_rate","avg_score","avg_ses",
                         "avg_attendance","risk_label"]].to_string())


# ════════════════════════════════════════════════════════════════════
# 4.  VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════
print("\n[4] Generating Visualizations...")

# ── Fig 1: Overview Dashboard ────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#F8F9FA")
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1A: Dropout by SES + COVID
ax = fig.add_subplot(gs[0, :2])
pivot = df.groupby(["ses_quintile","covid_year"])["dropout"].mean().unstack()
pivot.columns = ["Non-COVID Years","COVID Years (2563-64)"]
pivot.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[2]],
           edgecolor="white", width=0.7)
ax.set_title("Dropout Rate by SES × COVID Period", fontweight="bold", fontsize=12)
ax.set_xlabel("SES Quintile (1=Lowest)")
ax.set_ylabel("Dropout Rate")
ax.set_xticklabels([f"Q{i}" for i in range(1,6)], rotation=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1%}"))
ax.legend(framealpha=0.7)

# 1B: Score trend by year + SES
ax = fig.add_subplot(gs[0, 2:])
for q, color in zip([1, 3, 5], [PALETTE[3], PALETTE[0], PALETTE[2]]):
    grp = df[df["ses_quintile"]==q].groupby("academic_year")["avg_score"].mean()
    ax.plot(grp.index, grp.values, marker="o", ms=5, lw=2,
            color=color, label=f"Q{q}")
ax.axvspan(2563, 2564, alpha=0.12, color="orange", label="COVID")
ax.set_title("Average Score Trend 2557–2567 by SES", fontweight="bold", fontsize=12)
ax.set_xlabel("Academic Year"); ax.set_ylabel("Average Score")
ax.legend(title="SES Quintile", framealpha=0.7)

# 1C: ROC curves all models
ax = fig.add_subplot(gs[1, :2])
for (name, res), color in zip(results_m1.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y1_te, res["proba"])
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],"--", color="gray", lw=1)
ax.fill_between(fpr, tpr, alpha=0.06, color=PALETTE[1])
ax.set_title("ROC Curves — Dropout Early Warning", fontweight="bold", fontsize=12)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9, framealpha=0.7)

# 1D: Precision-Recall curve (best model)
ax = fig.add_subplot(gs[1, 2:])
prec_c, rec_c, _ = precision_recall_curve(y1_te, best_m1_prob)
ax.plot(rec_c, prec_c, lw=2.5, color=PALETTE[1])
ax.fill_between(rec_c, prec_c, alpha=0.15, color=PALETTE[1])
ax.axhline(y1_te.mean(), color="gray", ls="--", lw=1.2, label=f"Baseline ({y1_te.mean():.2%})")
ax.axvline(results_m1[best_m1_name]["ap"], color=PALETTE[2], ls=":", lw=1.5,
           label=f"AP={results_m1[best_m1_name]['ap']:.3f}")
ax.set_title(f"Precision-Recall — {best_m1_name}", fontweight="bold", fontsize=12)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(framealpha=0.7)

# 1E: Feature importance (dropout)
ax = fig.add_subplot(gs[2, :2])
top_fi1 = fi1.head(12)
bars = ax.barh(range(len(top_fi1)), top_fi1.values[::-1],
               color=[PALETTE[i % len(PALETTE)] for i in range(len(top_fi1))],
               edgecolor="white")
ax.set_yticks(range(len(top_fi1)))
ax.set_yticklabels(top_fi1.index[::-1], fontsize=9)
ax.set_title("Top Features — Dropout Model", fontweight="bold", fontsize=12)
ax.set_xlabel("Importance Score")

# 1F: Confusion matrix (best model at optimal threshold)
ax = fig.add_subplot(gs[2, 2:])
cm = confusion_matrix(y1_te, y1_pred_opt)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Dropout","Dropout"],
            yticklabels=["No Dropout","Dropout"],
            linewidths=0.5, annot_kws={"size":13})
ax.set_title(f"Confusion Matrix\n(thr={best_thr:.3f}, F2-optimized)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

fig.suptitle("Education ML Pipeline — Model Performance Dashboard",
             fontsize=16, fontweight="bold", y=1.01, color="#1a1a2e")
plt.savefig(f"{OUT_FIGS}/01_dashboard.png",
            bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
plt.close()
print("   → 01_dashboard.png")

# ── Fig 2: Score Model & Cluster Analysis ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.patch.set_facecolor("#F8F9FA")

# 2A: Actual vs Predicted reading
ax = axes[0,0]
ax.scatter(y2_te_r, pred_r, alpha=0.3, s=10, color=PALETTE[0])
mn, mx = min(y2_te_r.min(), pred_r.min()), max(y2_te_r.max(), pred_r.max())
ax.plot([mn,mx],[mn,mx], "r--", lw=1.5, label="Perfect")
ax.set_title("Reading Score: Actual vs Predicted", fontweight="bold")
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
ax.legend()

# 2B: Feature importance score
ax = axes[0,1]
top_fi2 = fi2.head(12)
ax.barh(range(len(top_fi2)), top_fi2.values[::-1],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(top_fi2))],
        edgecolor="white")
ax.set_yticks(range(len(top_fi2)))
ax.set_yticklabels(top_fi2.index[::-1], fontsize=9)
ax.set_title("Top Features — Score Model", fontweight="bold")
ax.set_xlabel("Importance")

# 2C: Residuals distribution
ax = axes[0,2]
residuals = y2_te_r.values - pred_r
ax.hist(residuals, bins=40, color=PALETTE[2], edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", lw=2, ls="--")
ax.set_title("Residual Distribution — Reading", fontweight="bold")
ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_ylabel("Count")
ax.text(0.97, 0.95, f"MAE={mean_absolute_error(y2_te_r,pred_r):.2f}\n"
        f"R²={r2_score(y2_te_r,pred_r):.3f}",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# 2D: Cluster scatter (PCA 2D)
ax = axes[1,0]
pca = PCA(n_components=2, random_state=42)
X3_pca = pca.fit_transform(X3_sc)
for ci, color in zip(range(best_k), PALETTE):
    mask = df["cluster"] == ci
    risk = cluster_profile.loc[ci, "risk_label"]
    ax.scatter(X3_pca[mask, 0], X3_pca[mask, 1],
               alpha=0.4, s=12, color=color,
               label=f"Cluster {ci} ({risk})")
ax.set_title("Student Clusters (PCA 2D)", fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.legend(fontsize=8, framealpha=0.7)

# 2E: Cluster profile heatmap (normalized)
ax = axes[1,1]
profile_cols = ["dropout_rate","avg_score","avg_ses","avg_attendance",
                "avg_internet","avg_distance","avg_policy_support"]
profile_norm = cluster_profile[profile_cols].copy()
for c in profile_cols:
    col_range = profile_norm[c].max() - profile_norm[c].min()
    profile_norm[c] = (profile_norm[c] - profile_norm[c].min()) / (col_range + 1e-9)

sns.heatmap(profile_norm, annot=cluster_profile[profile_cols].round(2),
            fmt="g", cmap="RdYlGn_r", ax=ax, linewidths=0.5,
            annot_kws={"size": 8})
ax.set_title("Cluster Profile Heatmap\n(normalized, red=high risk)",
             fontweight="bold")
ax.set_xticklabels(
    ["Dropout\nRate","Avg\nScore","SES","Attend-\nance",
     "Internet","Distance","Policy\nSupport"],
    rotation=30, ha="right", fontsize=8)
ax.set_yticklabels([f"C{i}" for i in cluster_profile.index], rotation=0)

# 2F: Silhouette + Elbow
ax = axes[1,2]
ax2b = ax.twinx()
ax.plot(list(K_RANGE), sil_scores, "o-", color=PALETTE[0], lw=2, label="Silhouette")
ax2b.plot(list(K_RANGE), inertias, "s--", color=PALETTE[2], lw=2, label="Inertia")
ax.axvline(best_k, color="red", lw=1.5, ls=":", label=f"Best k={best_k}")
ax.set_title("Silhouette & Elbow Curve", fontweight="bold")
ax.set_xlabel("Number of Clusters k")
ax.set_ylabel("Silhouette Score", color=PALETTE[0])
ax2b.set_ylabel("Inertia", color=PALETTE[2])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

fig.suptitle("Score Prediction & Student Clustering Analysis",
             fontsize=15, fontweight="bold", y=1.01, color="#1a1a2e")
plt.tight_layout()
plt.savefig(f"{OUT_FIGS}/02_score_cluster.png",
            bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
plt.close()
print("   → 02_score_cluster.png")

# ── Fig 3: Deployment & Policy Impact Simulation ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#F8F9FA")

# 3A: Policy intervention simulation (counterfactual)
ax = axes[0,0]
# simulate: ถ้าให้ทุกนักเรียน Q1-Q2 ได้รับ scholarship=1, internet_subsidy=1
df_sim = df.copy()
low_ses_mask = df_sim["ses_quintile"] <= 2
df_sim.loc[low_ses_mask, "scholarship"] = 1
df_sim.loc[low_ses_mask, "internet_subsidy"] = 1
df_sim.loc[low_ses_mask, "device_subsidy"] = 1
df_sim.loc[low_ses_mask, "policy_support_n"] += 1

X_sim = df_sim.loc[df.index, FEAT_DROPOUT].fillna(0)
prob_before = best_m1.predict_proba(X1)[:, 1]
prob_after  = best_m1.predict_proba(X_sim.loc[X1.index])[:, 1]

before_by_ses = df.groupby("ses_quintile").apply(
    lambda g: prob_before[g.index.intersection(pd.Index(range(len(prob_before))))].mean()
    if len(g.index.intersection(pd.Index(range(len(prob_before))))) > 0 else 0
)

# simpler: compare mean dropout prob before/after for each SES
X1_all = df[FEAT_DROPOUT].fillna(0)
X_sim_all = df_sim[FEAT_DROPOUT].fillna(0)
p_before = best_m1.predict_proba(X1_all)[:,1]
p_after  = best_m1.predict_proba(X_sim_all)[:,1]
df["p_dropout_before"] = p_before
df["p_dropout_after"]  = p_after

comp = df.groupby("ses_quintile")[["p_dropout_before","p_dropout_after"]].mean()
x = np.arange(5)
w = 0.35
bars_b = ax.bar(x - w/2, comp["p_dropout_before"], w, color=PALETTE[3],
                label="Before Policy", edgecolor="white")
bars_a = ax.bar(x + w/2, comp["p_dropout_after"], w, color=PALETTE[0],
                label="After Policy (sim)", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels([f"Q{i}" for i in range(1,6)])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1%}"))
ax.set_title("Policy Simulation: Universal Support for Q1-Q2\nPredicted Dropout Risk Before vs After",
             fontweight="bold")
ax.set_xlabel("SES Quintile"); ax.set_ylabel("Predicted Dropout Probability")
ax.legend(framealpha=0.7)

# annotate reduction
for i in range(5):
    diff = comp["p_dropout_after"].iloc[i] - comp["p_dropout_before"].iloc[i]
    if diff < -0.001:
        ax.text(i + w/2, comp["p_dropout_after"].iloc[i] + 0.001,
                f"{diff:.1%}", ha="center", va="bottom", fontsize=8,
                color=PALETTE[0], fontweight="bold")

# 3B: Score prediction by intervention
ax = axes[0,1]
df_q1 = df[df["ses_quintile"]==1].copy()
scenarios = {
    "Baseline"         : df_q1.copy(),
    "+Scholarship"     : df_q1.assign(scholarship=1, policy_support_n=df_q1["policy_support_n"]+1),
    "+Internet+Device" : df_q1.assign(internet_home=1, device_access=1,
                                       internet_subsidy=1, device_subsidy=1,
                                       digital_access=1,
                                       policy_support_n=df_q1["policy_support_n"]+2),
    "+Remedial"        : df_q1.assign(remedial_program=1,
                                       policy_support_n=df_q1["policy_support_n"]+1),
    "All Policies"     : df_q1.assign(scholarship=1, internet_home=1, device_access=1,
                                       internet_subsidy=1, device_subsidy=1,
                                       remedial_program=1, digital_access=1,
                                       policy_support_n=df_q1["policy_support_n"]+4),
}
pred_scores = {}
for scen, sdf in scenarios.items():
    X_sc = sdf[FEAT_SCORE].fillna(0)
    pred_scores[scen] = best_m2_r.predict(X_sc).mean()

colors_sc = [PALETTE[3], PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[4]]
bars = ax.barh(list(pred_scores.keys()), list(pred_scores.values()),
               color=colors_sc, edgecolor="white", height=0.6)
for bar, val in zip(bars, pred_scores.values()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontweight="bold", fontsize=10)
ax.set_xlim(0, max(pred_scores.values()) * 1.12)
ax.set_title("Predicted Reading Score (Q1 Students)\nUnder Different Policy Scenarios",
             fontweight="bold")
ax.set_xlabel("Predicted Avg Reading Score")

# 3C: Model calibration (reliability diagram)
ax = axes[1,0]
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins+1)
bin_mids, frac_pos, frac_pred = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (best_m1_prob >= lo) & (best_m1_prob < hi)
    if mask.sum() > 0:
        bin_mids.append((lo+hi)/2)
        frac_pos.append(y1_te[mask].mean())
        frac_pred.append(best_m1_prob[mask].mean())

ax.plot([0,1],[0,1],"--", color="gray", lw=1.5, label="Perfect calibration")
ax.plot(frac_pred, frac_pos, "o-", color=PALETTE[0], lw=2, ms=7,
        label=f"{best_m1_name}")
midpoints = [((lo+hi)/2) for lo,hi in zip(bin_edges[:-1],bin_edges[1:])]
ax.fill_between(frac_pred, frac_pos,
                midpoints[:len(frac_pred)],
                alpha=0.12, color=PALETTE[0])
ax.set_title("Calibration Curve (Reliability Diagram)\nDropout Early Warning Model",
             fontweight="bold")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (Actual)")
ax.legend(framealpha=0.7)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# 3D: Risk distribution in deployment (score threshold)
ax = axes[1,1]
thresholds_plot = [0.1, 0.2, 0.3, 0.4, 0.5]
flagged_pct = [(p_after >= t).mean() * 100 for t in thresholds_plot]
tp_est = [
    (p_after[df["dropout"]==1] >= t).mean() * df["dropout"].mean() * 100
    for t in thresholds_plot
]
ax.bar([f"≥{t:.0%}" for t in thresholds_plot], flagged_pct,
       color=PALETTE[3], alpha=0.7, label="% Flagged as At-Risk", edgecolor="white")
ax.bar([f"≥{t:.0%}" for t in thresholds_plot], tp_est,
       color=PALETTE[0], alpha=0.9, label="Est. True Dropout (caught)", edgecolor="white")
for i, (fp, tp) in enumerate(zip(flagged_pct, tp_est)):
    ax.text(i, fp + 0.3, f"{fp:.1f}%", ha="center", va="bottom",
            fontsize=8.5, color=PALETTE[3], fontweight="bold")
ax.set_title("Deployment Threshold Analysis\n% Students Flagged vs True Dropouts Caught",
             fontweight="bold")
ax.set_xlabel("Risk Threshold"); ax.set_ylabel("% of Total Students")
ax.legend(framealpha=0.7)

fig.suptitle("Policy Simulation & Deployment Analysis",
             fontsize=15, fontweight="bold", y=1.01, color="#1a1a2e")
plt.tight_layout()
plt.savefig(f"{OUT_FIGS}/03_deployment.png",
            bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
plt.close()
print("   → 03_deployment.png")


# ── Fig 4: Pipeline Architecture Diagram ────────────────────────
fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(0, 22); ax.set_ylim(0, 10)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color, fontsize=9.5, text_color="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15", linewidth=2,
                          edgecolor="white", facecolor=color, alpha=0.92)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            wrap=True, multialignment="center")

def draw_arrow(ax, x1, y1, x2, y2, color="white"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=16))

# Title
ax.text(11, 9.5, "Education ML Pipeline — End-to-End Architecture",
        ha="center", va="center", fontsize=16, color="white",
        fontweight="bold")

# Boxes
boxes = [
    # (x, y, w, h, text, color)
    (1.5, 7.5, 2.8, 1.2, "📂 Raw Data\nstudent_year_mechanism.csv\n(31K rows × 32 cols)", "#2E86AB"),
    (5.0, 7.5, 2.8, 1.2, "⚙️ Feature Engineering\n+Derived Features\n(vuln, digital, covid...)", "#A23B72"),
    (8.5, 7.5, 2.8, 1.2, "📊 EDA &\nData Validation\n(balance, drift)", "#F18F01"),
    (12.0, 7.5, 2.8, 1.2, "✂️ Train/Test Split\n80% / 20%\nStratified by dropout", "#44BBA4"),
    (15.5, 7.5, 2.8, 1.2, "⚖️ Class Imbalance\nHandling\n(sample_weight balanced)", "#C73E1D"),
    # Models
    (5.0, 5.0, 2.8, 1.2, "🎯 Model 1\nDropout Early Warning\n(GBM, RF, LR → Best AUC)", "#2E86AB"),
    (9.5, 5.0, 2.8, 1.2, "📈 Model 2\nScore Predictor\n(GBM Regressor)", "#A23B72"),
    (14.0, 5.0, 2.8, 1.2, "🔍 Model 3\nAt-Risk Clustering\n(K-Means + PCA)", "#F18F01"),
    # Evaluation
    (5.0, 3.0, 2.8, 1.2, "📋 Eval: AUC, AP\nF2-Score, Recall\nCalibration Curve", "#44BBA4"),
    (9.5, 3.0, 2.8, 1.2, "📋 Eval: MAE, RMSE\nR² Score\nResidual Analysis", "#44BBA4"),
    (14.0, 3.0, 2.8, 1.2, "📋 Eval: Silhouette\nElbow Curve\nCluster Profiling", "#44BBA4"),
    # Deployment
    (5.0, 1.2, 2.8, 1.1, "🚀 Deploy: REST API\n/predict/dropout\nFlask + joblib", "#E94F37"),
    (9.5, 1.2, 2.8, 1.1, "🚀 Deploy: REST API\n/predict/score\nInput → Predicted Score", "#E94F37"),
    (14.0, 1.2, 2.8, 1.1, "🚀 Dashboard\nAt-Risk List\nPriority Flag", "#E94F37"),
    (19.5, 5.0, 2.2, 5.5, "🔄 Policy\nSimulation\n\nCounterfactual\nAnalysis:\n\n• +Scholarship\n• +Internet\n• +Remedial\n\n→ Predicted\n  Impact", "#393E41"),
]

for bx, by, bw, bh, btxt, bcol in boxes:
    draw_box(ax, bx, by, bw, bh, btxt, bcol)

# Arrows horizontal (data pipeline)
for x1, x2 in [(2.9,3.6),(6.4,7.1),(9.9,10.6),(13.4,14.1)]:
    draw_arrow(ax, x1, 7.5, x2, 7.5)

# Arrows down to models
for mx in [5.0, 9.5, 14.0]:
    draw_arrow(ax, mx, 6.9, mx, 5.6)

# Arrows down to eval
for mx in [5.0, 9.5, 14.0]:
    draw_arrow(ax, mx, 4.4, mx, 3.6)

# Arrows down to deploy
for mx in [5.0, 9.5, 14.0]:
    draw_arrow(ax, mx, 2.4, mx, 1.75)

# Arrow: Feature Eng → Models (branch)
draw_arrow(ax, 3.7, 6.9, 5.0, 5.6)
draw_arrow(ax, 3.7, 6.9, 9.5, 5.6)
draw_arrow(ax, 3.7, 6.9, 14.0, 5.6)

# Arrow → policy sim
draw_arrow(ax, 18.1, 5.0, 18.5, 5.0)

plt.savefig(f"{OUT_FIGS}/04_pipeline_architecture.png",
            bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
plt.close()
print("   → 04_pipeline_architecture.png")


# ════════════════════════════════════════════════════════════════════
# 5.  SAVE MODELS & DEPLOYMENT ARTIFACTS
# ════════════════════════════════════════════════════════════════════
print("\n[5] Saving Models & Deployment Artifacts...")

joblib.dump(best_m1, f"{OUT_MODEL}/model1_dropout.pkl")
joblib.dump(best_m2_r, f"{OUT_MODEL}/model2_score_reading.pkl")
joblib.dump(best_m2_m, f"{OUT_MODEL}/model2_score_math.pkl")
joblib.dump(km_best, f"{OUT_MODEL}/model3_cluster.pkl")
joblib.dump(scaler3, f"{OUT_MODEL}/scaler_cluster.pkl")

metadata = {
    "model1": {
        "name": best_m1_name,
        "target": "dropout",
        "auc": round(results_m1[best_m1_name]["auc"], 4),
        "average_precision": round(results_m1[best_m1_name]["ap"], 4),
        "optimal_threshold": round(float(best_thr), 4),
        "features": FEAT_DROPOUT,
        "n_train": int(len(X1_tr)), "n_test": int(len(X1_te)),
    },
    "model2": {
        "name": "GradientBoostingRegressor",
        "target": "score_reading, score_math",
        "mae_reading": round(mean_absolute_error(y2_te_r, pred_r), 4),
        "r2_reading": round(r2_score(y2_te_r, pred_r), 4),
        "mae_math": round(mean_absolute_error(y2_te_m, pred_m), 4),
        "r2_math": round(r2_score(y2_te_m, pred_m), 4),
        "features": FEAT_SCORE,
    },
    "model3": {
        "name": "KMeans",
        "k": int(best_k),
        "silhouette": round(max(sil_scores), 4),
        "features": FEAT_CLUSTER,
        "cluster_risk_labels": {
            str(k): str(v)
            for k, v in cluster_profile["risk_label"].to_dict().items()
        },
    },
}
with open(f"{OUT_MODEL}/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"   → Models saved to {OUT_MODEL}/")


# ════════════════════════════════════════════════════════════════════
# 6.  DEPLOYMENT CODE (Flask API skeleton)
# ════════════════════════════════════════════════════════════════════
FLASK_CODE = '''"""
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
    d["grade_num"]        = int(\'\'\'
        str(d.get("grade_code","G1")).replace("G","").replace("V","")
    \'\'\'.strip()) if d.get("grade_code") else 1
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
'''

with open(f"{OUT_REPORT}/app.py", "w", encoding="utf-8") as f:
    f.write(FLASK_CODE)
print(f"   → app.py (Flask API) saved")


# ════════════════════════════════════════════════════════════════════
# 7.  PRINT FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MODEL 1: Dropout Early Warning                             ║
║  Algorithm : {best_m1_name:<44} ║
║  AUC       : {results_m1[best_m1_name]['auc']:.4f}                                        ║
║  Avg Prec  : {results_m1[best_m1_name]['ap']:.4f}                                        ║
║  Threshold : {best_thr:.4f} (F2-optimized, เน้น recall)          ║
║  Top feature: {fi1.index[0]:<43} ║
╠══════════════════════════════════════════════════════════════╣
║  MODEL 2: Score Predictor (GBM Regressor)                   ║
║  Reading → MAE={mean_absolute_error(y2_te_r,pred_r):.2f}, R²={r2_score(y2_te_r,pred_r):.4f}                     ║
║  Math    → MAE={mean_absolute_error(y2_te_m,pred_m):.2f}, R²={r2_score(y2_te_m,pred_m):.4f}                     ║
║  Top feature: {fi2.index[0]:<43} ║
╠══════════════════════════════════════════════════════════════╣
║  MODEL 3: At-Risk Clustering (K-Means)                      ║
║  Best k   : {best_k}                                              ║
║  Silhouette: {max(sil_scores):.4f}                                       ║
╠══════════════════════════════════════════════════════════════╣
║  DEPLOYMENT                                                  ║
║  → Flask API (app.py) ready                                  ║
║  → 4 endpoints: /predict/dropout, /predict/score,           ║
║                 /cluster/assign, /simulate/policy            ║
╚══════════════════════════════════════════════════════════════╝
""")

print(f"All outputs saved to: {OUT_REPORT}/")
print("  figures/ : 04 charts")
print("  models/  : pkl files + metadata.json")
print("  app.py   : Flask deployment API")
