"""Builds Multi_Disease_ML_Framework.ipynb from scratch using nbformat.
Run this once to (re)generate the notebook, then execute it with execute_notebook.py.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------- title ----
md("""# A Unified Explainable Ensemble Machine Learning Framework for Early Prediction of Multiple Chronic Diseases Using Routine Clinical Biomarkers

This notebook trains and explains the same ensemble learning pipeline (Random Forest, XGBoost, and a
stacked ensemble of both) separately on three public clinical datasets (Chronic Kidney Disease, Heart
Disease, and Diabetes), then compares results and feature importance across all three. Every chart,
table and dataset preview is saved to the project's `figures/` folder for use in the research paper.""")

# -------------------------------------------------------------- imports ----
code("""import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.getcwd())
from common import (
    class_balance_plot, correlation_heatmap, train_models,
    evaluate_models, run_shap, save_model, RESULTS,
    dataset_preview_image, summary_statistics_image, missing_values_plot,
    feature_distributions_plot, boxplots_by_class, all_metrics_bar_chart,
    builtin_feature_importance_plot, shap_dependence_plot, workflow_diagram,
)

BASE = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE, "data")
FIGURES_DIR = os.path.join(BASE, "figures")
MODELS_DIR = os.path.join(BASE, "notebook")
os.makedirs(FIGURES_DIR, exist_ok=True)
print("data:", DATA_DIR)
print("figures:", FIGURES_DIR)""")

md("## 0. Methodology Overview")
code("""workflow_diagram(FIGURES_DIR)""")


DISEASES = [
    {
        "name": "Chronic Kidney Disease",
        "intro": "## 1. Chronic Kidney Disease\n\n400 patients, 24 clinical and laboratory features. Source: UCI Machine Learning Repository (id=336).",
        "load": """ckd_raw = pd.read_csv(os.path.join(DATA_DIR, "ckd.csv"))
ckd_raw.columns = [c.strip() for c in ckd_raw.columns]
ckd_raw["class"] = ckd_raw["class"].astype(str).str.strip().replace({"ckd\\t": "ckd"})
print(ckd_raw.shape)
dataset_preview_image(ckd_raw, "Chronic Kidney Disease", FIGURES_DIR, stage="raw")
ckd_raw.head()""",
        "preclean_missing": """numeric_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

ckd_pre = ckd_raw.copy()
for c in numeric_cols:
    ckd_pre[c] = pd.to_numeric(ckd_pre[c], errors="coerce")
for c in categorical_cols:
    ckd_pre[c] = ckd_pre[c].astype(str).str.strip().replace({"nan": np.nan})
missing_values_plot(ckd_pre, "Chronic Kidney Disease", FIGURES_DIR)""",
        "clean": """ckd = ckd_pre.copy()
for c in numeric_cols:
    ckd[c] = ckd[c].fillna(ckd[c].median())
for c in categorical_cols:
    ckd[c] = ckd[c].fillna(ckd[c].mode()[0])
    ckd[c] = LabelEncoder().fit_transform(ckd[c])

ckd["target"] = (ckd["class"] == "ckd").astype(int)
ckd = ckd.drop(columns=["class"])
print(ckd.isnull().sum().sum(), "missing values remaining")
dataset_preview_image(ckd, "Chronic Kidney Disease", FIGURES_DIR, stage="cleaned")
summary_statistics_image(ckd, "Chronic Kidney Disease", FIGURES_DIR)
ckd.head()""",
        "eda": """class_balance_plot(ckd["target"], "Chronic Kidney Disease", FIGURES_DIR, positive_label="CKD")
correlation_heatmap(ckd, "Chronic Kidney Disease", FIGURES_DIR)
feature_distributions_plot(ckd, "Chronic Kidney Disease", FIGURES_DIR)

ckd_top_corr = ckd.drop(columns=["target"]).corrwith(ckd["target"]).abs().sort_values(ascending=False).head(4).index.tolist()
boxplots_by_class(ckd, "target", "Chronic Kidney Disease", FIGURES_DIR, ckd_top_corr, positive_label="CKD")
print("Top correlated features:", ckd_top_corr)""",
        "var": "ckd",
        "slug": "ckd",
    },
    {
        "name": "Heart Disease",
        "intro": "## 2. Heart Disease (Cleveland)\n\n303 patients, 13 clinical features. Source: UCI Machine Learning Repository (id=45). "
                 "The original target is 0-4 (severity); following standard practice for this dataset, it is binarized to presence (1) vs absence (0) of disease.",
        "load": """heart_raw = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
print(heart_raw.shape)
dataset_preview_image(heart_raw, "Heart Disease", FIGURES_DIR, stage="raw")
heart_raw.head()""",
        "preclean_missing": """heart_pre = heart_raw.copy()
for c in heart_pre.columns:
    heart_pre[c] = pd.to_numeric(heart_pre[c], errors="coerce")
missing_values_plot(heart_pre, "Heart Disease", FIGURES_DIR)""",
        "clean": """heart = heart_pre.copy()
heart["target"] = (heart["target"] > 0).astype(int)
for c in heart.columns:
    heart[c] = heart[c].fillna(heart[c].median())
dataset_preview_image(heart, "Heart Disease", FIGURES_DIR, stage="cleaned")
summary_statistics_image(heart, "Heart Disease", FIGURES_DIR)
heart.head()""",
        "eda": """class_balance_plot(heart["target"], "Heart Disease", FIGURES_DIR, positive_label="Disease Present")
correlation_heatmap(heart, "Heart Disease", FIGURES_DIR)
feature_distributions_plot(heart, "Heart Disease", FIGURES_DIR)

heart_top_corr = heart.drop(columns=["target"]).corrwith(heart["target"]).abs().sort_values(ascending=False).head(4).index.tolist()
boxplots_by_class(heart, "target", "Heart Disease", FIGURES_DIR, heart_top_corr, positive_label="Disease Present")
print("Top correlated features:", heart_top_corr)""",
        "var": "heart",
        "slug": "heart",
    },
    {
        "name": "Diabetes",
        "intro": "## 3. Diabetes\n\n768 patients, 8 clinical features. Source: Pima Indians Diabetes Database. "
                 "Five columns use 0 as a missing-value placeholder in the raw data (a well-documented quirk of this "
                 "dataset); those zeros are treated as missing and imputed with the median before training.",
        "load": """diabetes_raw = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
print(diabetes_raw.shape)
dataset_preview_image(diabetes_raw, "Diabetes", FIGURES_DIR, stage="raw")
diabetes_raw.head()""",
        "preclean_missing": """zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
diabetes_pre = diabetes_raw.copy()
for c in zero_as_missing:
    diabetes_pre[c] = diabetes_pre[c].replace(0, np.nan)
missing_values_plot(diabetes_pre, "Diabetes", FIGURES_DIR)""",
        "clean": """diabetes = diabetes_pre.copy()
for c in zero_as_missing:
    diabetes[c] = diabetes[c].fillna(diabetes[c].median())
diabetes = diabetes.rename(columns={"Outcome": "target"})
dataset_preview_image(diabetes, "Diabetes", FIGURES_DIR, stage="cleaned")
summary_statistics_image(diabetes, "Diabetes", FIGURES_DIR)
diabetes.head()""",
        "eda": """class_balance_plot(diabetes["target"], "Diabetes", FIGURES_DIR, positive_label="Diabetic")
correlation_heatmap(diabetes, "Diabetes", FIGURES_DIR)
feature_distributions_plot(diabetes, "Diabetes", FIGURES_DIR)

diabetes_top_corr = diabetes.drop(columns=["target"]).corrwith(diabetes["target"]).abs().sort_values(ascending=False).head(4).index.tolist()
boxplots_by_class(diabetes, "target", "Diabetes", FIGURES_DIR, diabetes_top_corr, positive_label="Diabetic")
print("Top correlated features:", diabetes_top_corr)""",
        "var": "diabetes",
        "slug": "diabetes",
    },
]

for d in DISEASES:
    v = d["var"]
    name = d["name"]
    md(d["intro"])
    code(d["load"])
    code(d["preclean_missing"])
    code(d["clean"])
    code(d["eda"])

    code(f"""X = {v}.drop(columns=["target"])
y = {v}["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

{v}_models = train_models(X_train, y_train)
{v}_best_name, {v}_best_model = evaluate_models({v}_models, X_test, y_test, "{name}", FIGURES_DIR)
all_metrics_bar_chart("{name}", FIGURES_DIR)
print("Best model:", {v}_best_name)""")

    code(f"""builtin_feature_importance_plot({v}_best_model, list(X.columns), "{name}", FIGURES_DIR, {v}_best_name)""")

    code(f"""{v}_ranking = run_shap({v}_best_model, X_train, X_test, "{name}", FIGURES_DIR,
                        model_kind="tree" if {v}_best_name != "Stacked Ensemble" else "other")
{v}_ranking.head(10)""")

    code(f"""{v}_top_feature = {v}_ranking.index[0]
shap_dependence_plot({v}_best_model, X_train, X_test, {v}_top_feature, "{name}", FIGURES_DIR,
                      model_kind="tree" if {v}_best_name != "Stacked Ensemble" else "other")""")

    code(f"""save_model({v}_best_model, "{name}", MODELS_DIR)
{v}_feature_order = list(X.columns)  # needed later by the Streamlit app""")

# =====================================================================
# CROSS-DISEASE COMPARISON
# =====================================================================
md("## 4. Cross-Disease Comparison\n\nThis is the section that turns three separate results into one "
   "study: a single results table across all nine model runs, and a look at which top SHAP features "
   "recur across diseases versus which are disease-specific.")

code("""results_df = pd.DataFrame(RESULTS)
results_df.to_csv(os.path.join(FIGURES_DIR, "..", "paper", "all_results.csv"), index=False)
results_df""")

code("""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
pivot = results_df.pivot(index="Model", columns="Disease", values="F1")
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("F1 score")
ax.set_title("F1 Score by Model and Disease")
ax.legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "cross_disease_f1_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out)""")

code("""top5 = pd.DataFrame({
    "Chronic Kidney Disease": ckd_ranking.head(5).index.tolist(),
    "Heart Disease": heart_ranking.head(5).index.tolist(),
    "Diabetes": diabetes_ranking.head(5).index.tolist(),
})
top5.to_csv(os.path.join(FIGURES_DIR, "..", "paper", "top5_features_per_disease.csv"), index=False)
top5""")

md("""### Notes for the paper

- `figures/` now holds, per disease: a raw and cleaned dataset preview table, a missing-values chart,
  summary statistics, class balance, correlation heatmap, feature distributions, boxplots by class,
  a full metrics comparison, built-in model feature importance, SHAP summary, and a SHAP dependence
  plot for the single most important feature, plus one methodology diagram and the cross-disease F1
  comparison.
- `paper/all_results.csv` holds every metric for all nine model runs (3 diseases x 3 models).
- `paper/top5_features_per_disease.csv` holds the top 5 SHAP-ranked features per disease.
- The three best-performing models are saved in `notebook/` as `.joblib` files for the Streamlit app.""")

nb["cells"] = cells
nbf.write(nb, "Multi_Disease_ML_Framework.ipynb")
print("notebook written: Multi_Disease_ML_Framework.ipynb,", len(cells), "cells")
