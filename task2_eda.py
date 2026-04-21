# =============================================================================
# COM747 2025-2026 — Data Science & Machine Learning Coursework
# Dataset : COPD GOLD 230 Patients (Kaggle)
# Task    : 2 — Exploratory Data Analysis (EDA) & Statistics
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEANED_DATA = "outputs/copd_cleaned.csv"  # output from Task 1
TARGET_COL   = "COPD GOLD"
OUTPUT_DIR   = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load cleaned data (run task1_preprocessing.py first) ─────────────────────
df = pd.read_csv(CLEANED_DATA)
print(f"Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

FEATURES     = [c for c in df.columns if c != TARGET_COL]
CLASS_LABELS = sorted(df[TARGET_COL].unique().tolist())

# =============================================================================
# TASK 2 — EDA & STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 65)
print("TASK 2 — EDA & STATISTICAL ANALYSIS")
print("=" * 65)

# ── 2.1 Descriptive statistics ────────────────────────────────────────────────
print("\n── Descriptive Statistics ──")
desc = df.describe().T
desc["skewness"] = df.skew()
desc["kurtosis"] = df.kurt()
print(desc.round(3).to_string())
desc.to_csv(f"{OUTPUT_DIR}/descriptive_statistics.csv")
print(f"\nDescriptive statistics saved → {OUTPUT_DIR}/descriptive_statistics.csv")

# ── 2.2 Class distribution ────────────────────────────────────────────────────
class_counts = df[TARGET_COL].value_counts().sort_index()
print("\n── Class Distribution ──")
print(class_counts)
print(f"Imbalance ratio (max/min): {class_counts.max()/class_counts.min():.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#1565C0", "#1E88E5", "#42A5F5", "#90CAF9"]
bars = ax.bar([f"GOLD {int(c)}" for c in CLASS_LABELS],
              class_counts.values, color=colors,
              edgecolor="black", linewidth=0.6)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3, str(val),
            ha="center", fontsize=10, fontweight="bold")
ax.set_title("COPD GOLD Stage Class Distribution", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Patients")
ax.set_xlabel("GOLD Stage")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"Class distribution chart saved → {OUTPUT_DIR}/class_distribution.png")

# ── 2.3 Feature distributions (histograms with KDE) ──────────────────────────
num_features = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
n_cols = 4
n_rows = (len(num_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(num_features):
    axes[i].hist(df[col], bins=20, color="#1E88E5",
                 edgecolor="white", linewidth=0.5, density=True, alpha=0.7)
    # Overlay KDE only if column has enough unique values (skip constants)
    unique_vals = df[col].dropna().nunique()
    if unique_vals > 5:
        try:
            kde_x = np.linspace(df[col].min(), df[col].max(), 200)
            kde = stats.gaussian_kde(df[col].dropna())
            axes[i].plot(kde_x, kde(kde_x), color="#E53935", lw=2)
        except Exception:
            pass  # skip KDE if it fails (e.g. near-constant columns)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=8)
    axes[i].set_ylabel("Density", fontsize=8)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions with KDE — COPD Dataset",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Feature distributions saved → {OUTPUT_DIR}/feature_distributions.png")

# ── 2.4 Correlation heatmap ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Correlation heatmap saved → {OUTPUT_DIR}/correlation_heatmap.png")

# ── 2.5 Boxplots by GOLD stage ────────────────────────────────────────────────
# Shows which features best separate GOLD stages visually
plot_features = num_features[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(plot_features):
    groups = [df[df[TARGET_COL] == cls][col].dropna().values
              for cls in CLASS_LABELS]
    bp = axes[i].boxplot(groups,
                         patch_artist=True,
                         labels=[f"GOLD {int(c)}" for c in CLASS_LABELS])
    colors_box = ["#BBDEFB", "#64B5F6", "#1E88E5", "#1565C0"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("GOLD Stage", fontsize=8)
    axes[i].tick_params(labelsize=7)

plt.suptitle("Feature Distributions by GOLD Stage",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/boxplots_by_gold_stage.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Boxplots saved → {OUTPUT_DIR}/boxplots_by_gold_stage.png")

# ── 2.6 Kruskal-Wallis test (statistical significance per feature) ────────────
# Non-parametric test for whether feature distributions differ across GOLD stages.
# Suitable here because we cannot assume normality in all features.
print("\n── Kruskal-Wallis Statistical Significance Test ──")
kw_results = []
for col in num_features:
    if col == TARGET_COL:
        continue
    groups = [df[df[TARGET_COL] == cls][col].dropna().values
              for cls in CLASS_LABELS]
    if all(len(g) > 0 for g in groups):
        stat, p = stats.kruskal(*groups)
        kw_results.append({
            "Feature": col,
            "H-statistic": round(stat, 4),
            "p-value": round(p, 4),
            "Significant (p<0.05)": "Yes" if p < 0.05 else "No"
        })

kw_df = pd.DataFrame(kw_results).sort_values("p-value")
print(kw_df.to_string(index=False))
kw_df.to_csv(f"{OUTPUT_DIR}/kruskal_wallis_results.csv", index=False)
print(f"\nKruskal-Wallis results saved → {OUTPUT_DIR}/kruskal_wallis_results.csv")

print("\n✓ Task 2 complete.")
