# =============================================================================
# COM747 2025-2026 — Data Science & Machine Learning Coursework
# Dataset : COPD GOLD 230 Patients (Kaggle)
# Task    : 3 — Baseline ML Classifiers
# Models  : Logistic Regression | Decision Tree | K-Nearest Neighbours
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEANED_DATA  = "outputs/copd_cleaned.csv"  # output from Task 1
TARGET_COL    = "COPD GOLD"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
N_FOLDS       = 5
OUTPUT_DIR    = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA & PREPARE TRAIN/TEST SPLIT
# =============================================================================

df = pd.read_csv(CLEANED_DATA)
print(f"Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
CLASS_LABELS = sorted(y.unique().tolist())

# ── Stratified 80/20 train/test split ─────────────────────────────────────────
# Stratification ensures each GOLD stage is proportionally represented
# in both the training and test sets — essential for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ── SMOTE: oversample minority GOLD stages (training set only) ────────────────
# SMOTE generates synthetic samples for minority classes to balance training.
# Applied ONLY to training data — never to the test set (avoids data leakage).
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_bal)} training samples")
print("Balanced class counts:\n",
      pd.Series(y_train_bal).value_counts().sort_index())

# ── Cross-validation strategy ─────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def confidence_interval(scores, z=1.96):
    """
    Compute mean and 95% confidence interval for cross-validation scores.

    Parameters
    ----------
    scores : array-like — CV accuracy scores per fold
    z      : float      — z-score for desired CI (1.96 = 95%)

    Returns
    -------
    mean, lower_bound, upper_bound : floats
    """
    mean = np.mean(scores)
    se   = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return mean, mean - z * se, mean + z * se


def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te, cv_strategy):
    """
    Train a sklearn Pipeline, cross-validate on training data, then
    evaluate on the held-out test set. Returns a full metrics dictionary.

    Using one reusable function (DRY principle) guarantees all models are
    evaluated with identical logic — avoiding inconsistencies.

    Parameters
    ----------
    name        : str      — display name
    pipeline    : Pipeline — sklearn Pipeline (scaler + classifier)
    X_tr, y_tr  : training features and labels
    X_te, y_te  : test features and labels
    cv_strategy : CV splitter

    Returns
    -------
    dict — all metrics + fitted pipeline + predictions + probabilities
    """
    # Cross-validate on training data only (no leakage)
    cv_scores = cross_val_score(pipeline, X_tr, y_tr,
                                cv=cv_strategy, scoring="accuracy")
    cv_mean, cv_lo, cv_hi = confidence_interval(cv_scores)

    # Fit on full training set then predict
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    # Core evaluation metrics (weighted = accounts for class imbalance)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    # ROC-AUC one-vs-rest multiclass
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
# TASK 3 — BASELINE CLASSIFIERS
# =============================================================================

print("\n" + "=" * 65)
print("TASK 3 — BASELINE CLASSIFIERS")
print("=" * 65)

# ── Logistic Regression ───────────────────────────────────────────────────────
# Justification: LR is a well-established probabilistic linear classifier
# providing interpretable coefficients and calibrated probability outputs,
# making it a natural first baseline in clinical classification tasks [1].
# class_weight='balanced' compensates for GOLD stage class imbalance.
# max_iter=1000 ensures convergence on this multi-class problem.
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
lr_res = evaluate_model("Logistic Regression",
                        lr_pipeline, X_train_bal, y_train_bal,
                        X_test, y_test, cv)

# ── Decision Tree ─────────────────────────────────────────────────────────────
# Justification: Decision Trees produce human-readable if-then rules,
# which is critical in clinical settings where clinician trust and
# explainability are required [2]. max_depth=5 regularises the tree
# to prevent overfitting on this relatively small dataset (n=230).
dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
dt_res = evaluate_model("Decision Tree",
                        dt_pipeline, X_train_bal, y_train_bal,
                        X_test, y_test, cv)

# ── K-Nearest Neighbours ──────────────────────────────────────────────────────
# Justification: KNN is a non-parametric instance-based learner making
# no assumptions about the data distribution — useful when decision
# boundaries between GOLD stages are complex and non-linear [3].
# Feature scaling is critical: KNN uses Euclidean distance, so unscaled
# features with large ranges would dominate the distance calculation.

# ── KNN Elbow Plot: find optimal k via cross-validation ──────────────────────
# Rather than assuming k=5, we empirically test k=1 to k=20 and select
# the value that maximises cross-validated F1 score.
print("  Finding optimal k for KNN via cross-validation elbow plot...")
k_range  = range(1, 21)
k_scores = []

for k in k_range:
    knn_temp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(knn_temp, X_train_bal, y_train_bal,
                             cv=cv, scoring="f1_weighted")
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"  Optimal k = {optimal_k}  (CV F1 = {max(k_scores):.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), k_scores, marker="o", color="#1565C0",
        linewidth=2, markersize=6)
ax.axvline(x=optimal_k, color="#E53935", linestyle="--",
           linewidth=1.5, label=f"Optimal k={optimal_k}")
ax.set_xlabel("Number of Neighbours (k)", fontsize=11)
ax.set_ylabel("CV F1-Score (Weighted)", fontsize=11)
ax.set_title("KNN — Elbow Plot for Optimal k Selection", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/knn_elbow_plot.png", dpi=150)
plt.close()
print(f"  KNN elbow plot saved → {OUTPUT_DIR}/knn_elbow_plot.png")

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=optimal_k))
])
knn_res = evaluate_model("K-Nearest Neighbours",
                         knn_pipeline, X_train_bal, y_train_bal,
                         X_test, y_test, cv)

# ── Save Task 3 results ───────────────────────────────────────────────────────
task3_results = [lr_res, dt_res, knn_res]
task3_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("pipeline", "y_pred", "y_prob")}
    for r in task3_results
])
task3_df.to_csv(f"{OUTPUT_DIR}/task3_baseline_results.csv", index=False)
print(f"\nTask 3 results saved → {OUTPUT_DIR}/task3_baseline_results.csv")

# ── Save test data and predictions for Task 5 to use ─────────────────────────
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test.values)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test.values)
pd.DataFrame({"Model": "Logistic Regression",
              "y_pred": lr_res["y_pred"]}).to_csv(
    f"{OUTPUT_DIR}/lr_predictions.csv", index=False)
pd.DataFrame({"Model": "Decision Tree",
              "y_pred": dt_res["y_pred"]}).to_csv(
    f"{OUTPUT_DIR}/dt_predictions.csv", index=False)
pd.DataFrame({"Model": "K-Nearest Neighbours",
              "y_pred": knn_res["y_pred"]}).to_csv(
    f"{OUTPUT_DIR}/knn_predictions.csv", index=False)

print("\n✓ Task 3 complete.")
