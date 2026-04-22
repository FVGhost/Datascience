# =============================================================================
# COM747 2025-2026 — Data Science & Machine Learning Coursework
# Dataset  : COPD GOLD 230 Patients (Kaggle)
# Task 5   : Evaluation, Benchmarking, Visualisation & Ethics
#
# CODE steps to work:
#   • Pipelines    → can be maintained  & no data leakage
#   • Functions    → able to be reused  (DRY principle)
#   • Constants    → make it frendly and updateable (one place to change)
#   • Docstrings   → making it readable  for the  collaborators
#   • RANDOM_STATE → reuse of all results
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

CLEANED_DATA  = "outputs/copd_cleaned.csv"
TARGET_COL    = "COPD GOLD"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
N_FOLDS       = 5
OUTPUT_DIR    = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA & PREPARE SPLIT
# Same RANDOM_STATE which is split to Tasks 3 & 4
# =============================================================================

df = pd.read_csv(CLEANED_DATA)
X  = df.drop(columns=[TARGET_COL])
y  = df[TARGET_COL]
CLASS_LABELS = sorted(y.unique().tolist())
FEATURES     = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def confidence_interval(scores, z=1.96):
    """
    Compute mean and 95% confidence interval for CV scores.

    Parameters
    ----------
    scores : array-like — per-fold CV scores
    z      : float      — z-score (1.96 = 95% CI)

    Returns
    -------
    mean, lower, upper : floats
    """
    mean = np.mean(scores)
    se   = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return mean, mean - z * se, mean + z * se


def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te, cv_strategy):
    """
    Train a Pipeline, cross-validate on training data, evaluate on test set.

    Reusable function (DRY principle) guarantees identical evaluation logic
    across all five models, ensuring fair and consistent benchmarking.

    Parameters
    ----------
    name        : str      — model display name
    pipeline    : Pipeline — sklearn Pipeline (scaler + classifier)
    X_tr, y_tr  : training features and labels
    X_te, y_te  : test features and labels
    cv_strategy : CV splitter

    Returns
    -------
    dict of metrics + fitted pipeline + predictions + probabilities
    """
    cv_scores = cross_val_score(pipeline, X_tr, y_tr,
                                cv=cv_strategy, scoring="accuracy")
    cv_mean, cv_lo, cv_hi = confidence_interval(cv_scores)

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    try:
        y_prob = pipeline.predict_proba(X_te)
        auc    = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        y_prob, auc = None, None

    return {
        "Model"         : name,
        "CV Accuracy"   : round(cv_mean, 4),
        "CV CI Low"     : round(cv_lo, 4),
        "CV CI High"    : round(cv_hi, 4),
        "Test Accuracy" : round(acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall"   : round(rec, 4),
        "Test F1"       : round(f1, 4),
        "Test AUC"      : round(auc, 4) if auc else "N/A",
        "pipeline"      : pipeline,
        "y_pred"        : y_pred,
        "y_prob"        : y_prob,
    }

# =============================================================================
# RE-TRAIN ALL MODELS
# Task 5 refresh everything so it can produce  accurate evaluation outputs.
# =============================================================================

print("Re-training all models for combined evaluation...")

# ── Logistic Regression ───────────────────────────────────────────────────────
lr_res = evaluate_model("Logistic Regression",
    Pipeline([("scaler", StandardScaler()),
              ("clf", LogisticRegression(max_iter=1000,
               class_weight="balanced", random_state=RANDOM_STATE))]),
    X_train_bal, y_train_bal, X_test, y_test, cv)

# ── Decision Tree ─────────────────────────────────────────────────────────────
dt_res = evaluate_model("Decision Tree",
    Pipeline([("scaler", StandardScaler()),
              ("clf", DecisionTreeClassifier(max_depth=5,
               class_weight="balanced", random_state=RANDOM_STATE))]),
    X_train_bal, y_train_bal, X_test, y_test, cv)

# ── KNN with optimal k (elbow method from Task 3) ────────────────────────────
k_range  = range(1, 21)
k_scores = []
for k in k_range:
    knn_temp = Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=k))])
    scores = cross_val_score(knn_temp, X_train_bal, y_train_bal,
                             cv=cv, scoring="f1_weighted")
    k_scores.append(scores.mean())
optimal_k = list(k_range)[np.argmax(k_scores)]
print(f"  KNN optimal k = {optimal_k}")

knn_res = evaluate_model("K-Nearest Neighbours",
    Pipeline([("scaler", StandardScaler()),
              ("clf", KNeighborsClassifier(n_neighbors=optimal_k))]),
    X_train_bal, y_train_bal, X_test, y_test, cv)

# ── Random Forest default ─────────────────────────────────────────────────────
rf_base_res = evaluate_model("Random Forest (Default)",
    Pipeline([("scaler", StandardScaler()),
              ("clf", RandomForestClassifier(class_weight="balanced",
               random_state=RANDOM_STATE))]),
    X_train_bal, y_train_bal, X_test, y_test, cv)

# ── Random Forest tuned via GridSearchCV ─────────────────────────────────────
print("  Running GridSearchCV...")
rf_grid = GridSearchCV(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced",
                                       random_state=RANDOM_STATE))
    ]),
    param_grid={
        "clf__n_estimators"      : [100, 200, 300],
        "clf__max_depth"         : [None, 5, 10, 15],
        "clf__min_samples_split" : [2, 5, 10],
        "clf__max_features"      : ["sqrt", "log2"],
    },
    cv=cv, scoring="f1_weighted",
    n_jobs=-1, verbose=0, refit=True
)
rf_grid.fit(X_train_bal, y_train_bal)
print(f"  Best RF params: {rf_grid.best_params_}")

rf_tuned_res = evaluate_model("Random Forest (Tuned)",
    rf_grid.best_estimator_,
    X_train_bal, y_train_bal, X_test, y_test, cv)

all_results = [lr_res, dt_res, knn_res, rf_base_res, rf_tuned_res]

# =============================================================================
# TASK 5 — FULL EVALUATION & BENCHMARKING
# =============================================================================

print("\n" + "=" * 65)
print("TASK 5 — EVALUATION, BENCHMARKING & VISUALISATION")
print("=" * 65)

# ── 5.1 Full benchmarking table ───────────────────────────────────────────────
comparison_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("pipeline", "y_pred", "y_prob")}
    for r in all_results
])
print("\nFull Benchmarking Table:\n")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{OUTPUT_DIR}/benchmarking_results.csv", index=False)
print(f"\nBenchmarking table saved → {OUTPUT_DIR}/benchmarking_results.csv")

# ── 5.2 Tuning improvement summary ───────────────────────────────────────────
print("\n── Tuning Improvement Summary ──")
rf_default_f1  = float(comparison_df.loc[comparison_df["Model"] == "Random Forest (Default)", "Test F1"].values[0])
rf_tuned_f1    = float(comparison_df.loc[comparison_df["Model"] == "Random Forest (Tuned)",   "Test F1"].values[0])
rf_default_auc = float(comparison_df.loc[comparison_df["Model"] == "Random Forest (Default)", "Test AUC"].values[0])
rf_tuned_auc   = float(comparison_df.loc[comparison_df["Model"] == "Random Forest (Tuned)",   "Test AUC"].values[0])

f1_gain  = (rf_tuned_f1  - rf_default_f1)  * 100
auc_gain = (rf_tuned_auc - rf_default_auc) * 100

baseline_mask = comparison_df["Model"].isin(
    ["Logistic Regression", "Decision Tree", "K-Nearest Neighbours"]
)
best_baseline_f1 = comparison_df[baseline_mask]["Test F1"].astype(float).max()
best_baseline_nm = comparison_df.loc[
    comparison_df["Test F1"].astype(float) == best_baseline_f1, "Model"
].values[0]
vs_baseline = (rf_tuned_f1 - best_baseline_f1) * 100

print(f"  RF Default F1  : {rf_default_f1:.4f}")
print(f"  RF Tuned F1    : {rf_tuned_f1:.4f}  ({f1_gain:+.2f}% improvement)")
print(f"  RF Default AUC : {rf_default_auc:.4f}")
print(f"  RF Tuned AUC   : {rf_tuned_auc:.4f}  ({auc_gain:+.2f}% improvement)")
print(f"  Tuned RF vs best baseline ({best_baseline_nm}): {vs_baseline:+.2f}% F1")

summary_lines = [
    f"RF Default F1  : {rf_default_f1:.4f}",
    f"RF Tuned F1    : {rf_tuned_f1:.4f}  ({f1_gain:+.2f}% improvement)",
    f"RF Default AUC : {rf_default_auc:.4f}",
    f"RF Tuned AUC   : {rf_tuned_auc:.4f}  ({auc_gain:+.2f}% improvement)",
    f"Tuned RF vs best baseline ({best_baseline_nm}): {vs_baseline:+.2f}% F1",
]
with open(f"{OUTPUT_DIR}/improvement_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
print(f"Improvement summary saved → {OUTPUT_DIR}/improvement_summary.txt")

# ── 5.3 Confusion matrices (all models) ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, result in enumerate(all_results):
    cm   = confusion_matrix(y_test, result["y_pred"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"GOLD {int(c)}" for c in CLASS_LABELS]
    )
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(result["Model"], fontsize=10, fontweight="bold")
    axes[i].tick_params(axis="x", rotation=30, labelsize=8)

axes[-1].set_visible(False)
fig.suptitle("Confusion Matrices — All Models (Test Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrices saved → {OUTPUT_DIR}/confusion_matrices.png")

# ── 5.4 Benchmarking bar chart (F1 + CV accuracy with 95% CI) ────────────────
fig, ax = plt.subplots(figsize=(11, 5))
models   = comparison_df["Model"]
f1_vals  = comparison_df["Test F1"].astype(float)
cv_means = comparison_df["CV Accuracy"].astype(float)
cv_lo    = comparison_df["CV CI Low"].astype(float)
cv_hi    = comparison_df["CV CI High"].astype(float)
ci_err   = [cv_means - cv_lo, cv_hi - cv_means]

x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, f1_vals, w,
               label="Test F1 (Weighted)",
               color="#1565C0", edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + w/2, cv_means, w,
               label="CV Accuracy (Mean ± 95% CI)",
               color="#2E7D32", edgecolor="black", linewidth=0.6,
               yerr=ci_err, capsize=5, error_kw={"elinewidth": 1.5})

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Benchmarking — Test F1 vs CV Accuracy with 95% CI",
             fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
ax.set_ylim(0, 1.18)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.04,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Comparison chart saved → {OUTPUT_DIR}/model_comparison_chart.png")

y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
colors_roc = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]

fig, ax = plt.subplots(figsize=(7, 6))
for i, cls in enumerate(sorted(y.unique())):
    if rf_tuned_res["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_tuned_res["y_prob"][:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], rf_tuned_res["y_prob"][:, i])
        ax.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)],
                lw=2, label=f"GOLD {int(cls)}  (AUC = {auc_i:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title("ROC Curves — Random Forest (Tuned), One-vs-Rest",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"ROC curves saved → {OUTPUT_DIR}/roc_curves.png")

# ── 5.5 Feature importances with error bars ──────────────────────────
rf_clf      = rf_grid.best_estimator_.named_steps["clf"]
importances = pd.Series(rf_clf.feature_importances_, index=FEATURES)
std_imp     = np.std([t.feature_importances_ for t in rf_clf.estimators_], axis=0)
std_series  = pd.Series(std_imp, index=FEATURES)
importances = importances.sort_values(ascending=False)
std_series  = std_series[importances.index]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(importances.index, importances.values,
       yerr=std_series.values,
       color="#1565C0", edgecolor="black", linewidth=0.5,
       capsize=3, error_kw={"elinewidth": 1.2})
ax.set_title("Random Forest — Feature Importances (Tuned) ± Std Dev",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Decrease in Impurity")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Feature importances saved → {OUTPUT_DIR}/feature_importances.png")


print("\nGenerating learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    rf_grid.best_estimator_,
    X_train_bal, y_train_bal,
    cv=cv,
    scoring="f1_weighted",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_mean, "o-", color="#1565C0",
        linewidth=2, label="Training F1")
ax.fill_between(train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.15, color="#1565C0")
ax.plot(train_sizes, val_mean, "o-", color="#E53935",
        linewidth=2, label="Cross-Validation F1")
ax.fill_between(train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.15, color="#E53935")
ax.set_xlabel("Training Set Size", fontsize=11)
ax.set_ylabel("F1-Score (Weighted)", fontsize=11)
ax.set_title("Learning Curve — Random Forest (Tuned)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Learning curve saved → {OUTPUT_DIR}/learning_curve.png")

# ── 5.7 Per-class F1 heatmap (all models) ────────────────────────────────────
per_class_f1 = {}
for result in all_results:
    report = classification_report(
        y_test, result["y_pred"], output_dict=True, zero_division=0
    )
    per_class_f1[result["Model"]] = {
        f"GOLD {int(cls)}": report.get(str(cls), {}).get("f1-score", 0)
        for cls in CLASS_LABELS
    }

heatmap_df = pd.DataFrame(per_class_f1).T
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Per-Class F1-Score — All Models vs GOLD Stage",
             fontsize=11, fontweight="bold")
ax.set_xlabel("GOLD Stage")
ax.set_ylabel("Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_f1_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Per-class F1 heatmap saved → {OUTPUT_DIR}/per_class_f1_heatmap.png")

# ── 5.8 Final summary ─────────────────────────────────────────────────────────
best_model = comparison_df.loc[
    comparison_df["Test F1"].astype(float).idxmax(), "Model"
]
best_f1  = comparison_df["Test F1"].astype(float).max()
best_auc_row = comparison_df[comparison_df["Test AUC"] != "N/A"]
best_auc = best_auc_row["Test AUC"].astype(float).max() if not best_auc_row.empty else "N/A"

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"Best model (Test F1): {best_model}")
print(f"Best Test F1        : {best_f1:.4f}")
print(f"Best Test AUC       : {best_auc}")

print("\n✓ Task 5 complete.")
