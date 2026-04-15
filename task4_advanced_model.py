# =============================================================================
# COM747 2025-2026 — Data Science & Machine Learning Coursework
# Dataset : COPD GOLD 230 Patients (Kaggle)
# Task    : 4 — Advanced Model + Tuning
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEANED_DATA  = "outputs/copd_cleaned.csv"
TARGET_COL    = "COPD GOLD"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
N_FOLDS       = 5
OUTPUT_DIR    = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA & PREPARE TRAIN/TEST SPLIT
# (must match exact split from Task 3 — same RANDOM_STATE and stratify)
# =============================================================================

df = pd.read_csv(CLEANED_DATA)
X  = df.drop(columns=[TARGET_COL])
y  = df[TARGET_COL]
CLASS_LABELS = sorted(y.unique().tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y
)

# Apply SMOTE to training set only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Train (balanced): {len(X_train_bal)}  |  Test: {len(X_test)}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def confidence_interval(scores, z=1.96):
    """
    Compute mean and 95% confidence interval for cross-validation scores.

    Parameters
    ----------
    scores : array-like — per-fold CV scores
    z      : float      — z-score (1.96 for 95% CI)

    Returns
    -------
    mean, lower_bound, upper_bound : floats
    """
    mean = np.mean(scores)
    se   = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return mean, mean - z * se, mean + z * se


def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te, cv_strategy):
    """
    Train, cross-validate, and evaluate a sklearn Pipeline.

    Reusable function ensures consistent, comparable evaluation
    across all models (DRY principle).

    Parameters
    ----------
    name        : str      — model display name
    pipeline    : Pipeline — sklearn Pipeline
    X_tr, y_tr  : training data
    X_te, y_te  : test data
    cv_strategy : CV splitter

    Returns
    -------
    dict of all metrics + fitted pipeline + predictions + probabilities
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

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  CV Accuracy  : {cv_mean:.4f}  95% CI [{cv_lo:.4f}, {cv_hi:.4f}]")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Precision    : {prec:.4f}  (weighted)")
    print(f"  Recall       : {rec:.4f}  (weighted)")
    print(f"  F1-Score     : {f1:.4f}  (weighted)")
    if auc:
        print(f"  ROC-AUC      : {auc:.4f}  (weighted OvR)")
    print(f"\n  Per-class Report:\n"
          + classification_report(y_te, y_pred, zero_division=0))

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
# TASK 4 — RANDOM FOREST + GRIDSEARCHCV TUNING
# =============================================================================

print("\n" + "=" * 65)
print("TASK 4 — RANDOM FOREST + HYPERPARAMETER TUNING")
print("=" * 65)

# ── 4a. Random Forest — default (pre-tuning baseline) ────────────────────────
# Justification: Random Forest is a bagging ensemble of decorrelated decision
# trees. It reduces variance compared to a single Decision Tree, handles
# mixed feature types naturally, and consistently outperforms single
# classifiers on medical classification tasks [4].
# class_weight='balanced' corrects for GOLD stage class imbalance.
rf_base = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
rf_base_res = evaluate_model("Random Forest (Default)",
                             rf_base, X_train_bal, y_train_bal,
                             X_test, y_test, cv)

# ── 4b. GridSearchCV hyperparameter tuning ────────────────────────────────────
# Exhaustive search over the most impactful Random Forest hyperparameters:
#
#   n_estimators     — number of trees. More trees reduce variance but
#                      have diminishing returns beyond ~300.
#   max_depth        — maximum tree depth. None = fully grown (may overfit).
#                      Limiting depth regularises the model.
#   min_samples_split— minimum samples required to split a node.
#                      Higher values create smoother decision boundaries.
#   max_features     — number of features considered at each split.
#                      'sqrt' is the standard recommendation for classifiers.
#
# Scoring: f1_weighted optimises for F1 across all GOLD stages,
# accounting for class imbalance rather than just majority class accuracy.

print("\nRunning GridSearchCV — please wait...")

param_grid = {
    "clf__n_estimators"      : [100, 200, 300],
    "clf__max_depth"         : [None, 5, 10, 15],
    "clf__min_samples_split" : [2, 5, 10],
    "clf__max_features"      : ["sqrt", "log2"],
}

rf_grid = GridSearchCV(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),
    param_grid=param_grid,
    cv=cv,
    scoring="f1_weighted",  # imbalance-aware scoring metric
    n_jobs=-1,              # use all CPU cores for speed
    verbose=1,
    refit=True              # refit best model on full training set
)

rf_grid.fit(X_train_bal, y_train_bal)

print(f"\nBest hyperparameters : {rf_grid.best_params_}")
print(f"Best CV F1 (weighted): {rf_grid.best_score_:.4f}")

rf_tuned_res = evaluate_model("Random Forest (Tuned)",
                              rf_grid.best_estimator_,
                              X_train_bal, y_train_bal,
                              X_test, y_test, cv)

# ── Save GridSearch top 10 configurations ─────────────────────────────────────
cv_results_df = pd.DataFrame(rf_grid.cv_results_)
top10 = (cv_results_df
         .sort_values("mean_test_score", ascending=False)
         .head(10)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
         .reset_index(drop=True))
print("\nTop 10 GridSearch configurations:\n", top10.to_string())
top10.to_csv(f"{OUTPUT_DIR}/gridsearch_top10.csv", index=False)
print(f"\nGridSearch results saved → {OUTPUT_DIR}/gridsearch_top10.csv")

# ── Save Task 4 results ───────────────────────────────────────────────────────
task4_results = [rf_base_res, rf_tuned_res]
task4_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("pipeline", "y_pred", "y_prob")}
    for r in task4_results
])
task4_df.to_csv(f"{OUTPUT_DIR}/task4_rf_results.csv", index=False)
print(f"Task 4 results saved → {OUTPUT_DIR}/task4_rf_results.csv")

# Save tuned RF predictions for Task 5
pd.DataFrame({"Model": "Random Forest (Default)",
              "y_pred": rf_base_res["y_pred"]}).to_csv(
    f"{OUTPUT_DIR}/rf_base_predictions.csv", index=False)
pd.DataFrame({"Model": "Random Forest (Tuned)",
              "y_pred": rf_tuned_res["y_pred"]}).to_csv(
    f"{OUTPUT_DIR}/rf_tuned_predictions.csv", index=False)

if rf_tuned_res["y_prob"] is not None:
    np.save(f"{OUTPUT_DIR}/rf_tuned_proba.npy", rf_tuned_res["y_prob"])

# ── Learning curve: diagnose overfitting vs underfitting ─────────────────────
# A learning curve plots training and CV validation accuracy as a function
# of training set size. It reveals whether the model suffers from:
#   - High bias (underfitting): both curves plateau at low accuracy
#   - High variance (overfitting): large gap between train and CV curves
# This is a standard experiment design technique cited in the rubric.
from sklearn.model_selection import learning_curve

print("\nGenerating learning curve for tuned Random Forest...")
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
ax.fill_between(train_sizes, train_mean - train_std,
                train_mean + train_std, alpha=0.15, color="#1565C0")
ax.plot(train_sizes, val_mean, "o-", color="#E53935",
        linewidth=2, label="Cross-Validation F1")
ax.fill_between(train_sizes, val_mean - val_std,
                val_mean + val_std, alpha=0.15, color="#E53935")
ax.set_xlabel("Training Set Size", fontsize=11)
ax.set_ylabel("F1-Score (Weighted)", fontsize=11)
ax.set_title("Learning Curve — Random Forest (Tuned)", fontsize=11,
             fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/learning_curve.png", dpi=150)
plt.close()
print(f"Learning curve saved → {OUTPUT_DIR}/learning_curve.png")

# ── Save feature importances for Task 5
rf_clf      = rf_grid.best_estimator_.named_steps["clf"]
feat_imp_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": rf_clf.feature_importances_,
    "Std Dev"   : np.std([t.feature_importances_ for t in rf_clf.estimators_], axis=0)
}).sort_values("Importance", ascending=False)
feat_imp_df.to_csv(f"{OUTPUT_DIR}/feature_importances.csv", index=False)
print(f"Feature importances saved → {OUTPUT_DIR}/feature_importances.csv")

print("\n✓ Task 4 complete.")
