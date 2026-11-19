"""Shared training, evaluation and explainability pipeline used for all three
diseases in this study. Keeping this logic in one place is what makes the
notebook a genuine unified framework rather than three separate scripts that
happen to sit next to each other.
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import shap.explainers._tree as _shap_tree
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")

RESULTS = []  # one row per (disease, model) — feeds the cross-disease comparison

# --- compatibility fix -----------------------------------------------------
# XGBoost 3.x stores "base_score" in its raw model dump as a bracketed
# single-element array string (e.g. "[5E-1]") instead of a plain number.
# shap 0.49.1's model loader calls float() on it directly and crashes. This
# patch normalizes that one field right after shap decodes the raw model,
# before its own parsing code touches it — nothing else is changed.
_original_decode_ubjson = _shap_tree.decode_ubjson_buffer


def _patched_decode_ubjson_buffer(fd):
    jmodel = _original_decode_ubjson(fd)
    try:
        lmp = jmodel["learner"]["learner_model_param"]
        bs = lmp.get("base_score")
        if isinstance(bs, str) and bs.startswith("["):
            lmp["base_score"] = bs.strip("[]")
    except (KeyError, TypeError):
        pass
    return jmodel


_shap_tree.decode_ubjson_buffer = _patched_decode_ubjson_buffer


def class_balance_plot(y, disease_name, figures_dir, positive_label="Positive"):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Negative", positive_label] if len(counts) == 2 else counts.index.astype(str)
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_title(f"{disease_name}: Class Balance")
    ax.set_ylabel("Number of patients")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{disease_name.lower().replace(' ', '_')}_class_balance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def correlation_heatmap(df, disease_name, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
    ax.set_title(f"{disease_name}: Feature Correlation Heatmap")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{disease_name.lower().replace(' ', '_')}_correlation_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def train_models(X_train, y_train, random_state=42):
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=random_state,
    )
    xgb.fit(X_train, y_train)

    stack = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
                    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           eval_metric="logloss", random_state=random_state))],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
    stack.fit(X_train, y_train)

    return {"Random Forest": rf, "XGBoost": xgb, "Stacked Ensemble": stack}


def evaluate_models(models, X_test, y_test, disease_name, figures_dir):
    best_name, best_model, best_f1 = None, None, -1
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        RESULTS.append({
            "Disease": disease_name, "Model": name, "Accuracy": round(acc, 3),
            "Precision": round(prec, 3), "Recall": round(rec, 3),
            "F1": round(f1, 3), "ROC-AUC": round(auc, 3),
        })
        print(f"{disease_name} / {name}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    # confusion matrix + ROC curve for the best model only (that's the one the paper discusses in depth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0], cmap="Blues")
    axes[0].set_title(f"{disease_name}: Confusion Matrix ({best_name})")
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"{disease_name}: ROC Curve ({best_name})")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{disease_name.lower().replace(' ', '_')}_confusion_roc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)

    return best_name, best_model


def run_shap(model, X_train, X_test, disease_name, figures_dir, model_kind="tree"):
    if model_kind == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            # older SHAP API: one array per class
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # newer SHAP API: single array shaped (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.Explainer(model.predict_proba, X_train, algorithm="permutation")
        shap_values = explainer(X_test).values[:, :, 1]

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{disease_name}: SHAP Feature Importance")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{disease_name.lower().replace(' ', '_')}_shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    return ranking


def _slug(disease_name):
    return disease_name.lower().replace(" ", "_")


def dataset_preview_image(df, disease_name, figures_dir, stage, n=8, max_cols=14):
    shown = df.iloc[:n, :max_cols].copy()
    for c in shown.columns:
        if shown[c].dtype == float:
            shown[c] = shown[c].round(2)
    fig, ax = plt.subplots(figsize=(min(1.1 * len(shown.columns), 18), 0.5 * (n + 2)))
    ax.axis("off")
    note = "" if df.shape[1] <= max_cols else f"  (showing first {max_cols} of {df.shape[1]} columns)"
    ax.set_title(f"{disease_name}: Sample of the {stage} dataset{note}", fontsize=11, pad=14)
    tbl = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_dataset_preview_{stage}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def summary_statistics_image(df, disease_name, figures_dir):
    desc = df.describe().round(2).reset_index().rename(columns={"index": "statistic"})
    max_cols = 10
    shown = desc.iloc[:, :max_cols]
    fig, ax = plt.subplots(figsize=(min(1.3 * len(shown.columns), 16), 0.45 * (len(shown) + 2)))
    ax.axis("off")
    note = "" if desc.shape[1] <= max_cols else f"  (showing first {max_cols - 1} of {desc.shape[1] - 1} features)"
    ax.set_title(f"{disease_name}: Summary Statistics{note}", fontsize=11, pad=14)
    tbl = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_summary_statistics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def missing_values_plot(df, disease_name, figures_dir):
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(nulls) + 1)))
    if len(nulls) == 0:
        ax.text(0.5, 0.5, "No missing values in this raw dataset", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.barh(nulls.index[::-1], nulls.values[::-1], color="#C44E52")
        ax.set_xlabel("Number of missing values")
        for i, v in enumerate(nulls.values[::-1]):
            ax.text(v + max(nulls.values) * 0.01, i, str(v), va="center")
    ax.set_title(f"{disease_name}: Missing Values by Feature (raw data)")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_missing_values.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def feature_distributions_plot(df, disease_name, figures_dir, exclude=("target",)):
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude if c in df.columns])
    n = len(numeric.columns)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric.columns):
        axes[i].hist(numeric[col].dropna(), bins=20, color="#4C72B0", edgecolor="white")
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{disease_name}: Feature Distributions", fontsize=13)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_feature_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def boxplots_by_class(df, target_col, disease_name, figures_dir, top_features, positive_label="Positive"):
    n = len(top_features)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, top_features):
        data = [df.loc[df[target_col] == 0, col], df.loc[df[target_col] == 1, col]]
        ax.boxplot(data, tick_labels=["Negative", positive_label])
        ax.set_title(col, fontsize=10)
    fig.suptitle(f"{disease_name}: Top Features by Outcome Class", fontsize=13)
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_boxplots_by_class.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def all_metrics_bar_chart(disease_name, figures_dir):
    df = pd.DataFrame([r for r in RESULTS if r["Disease"] == disease_name])
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(9, 5))
    df.set_index("Model")[metrics].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{disease_name}: All Metrics by Model")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_all_metrics_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def builtin_feature_importance_plot(model, feature_names, disease_name, figures_dir, model_name):
    if not hasattr(model, "feature_importances_"):
        print(f"skipped built-in importance plot for {disease_name} ({model_name} has no feature_importances_)")
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(f"{disease_name}: {model_name} Built-in Feature Importance")
    fig.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_builtin_feature_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def shap_dependence_plot(model, X_train, X_test, top_feature, disease_name, figures_dir, model_kind="tree"):
    if model_kind == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.Explainer(model.predict_proba, X_train, algorithm="permutation")
        shap_values = explainer(X_test).values[:, :, 1]

    fig = plt.figure(figsize=(7, 5))
    shap.dependence_plot(top_feature, shap_values, X_test, show=False, ax=plt.gca())
    plt.title(f"{disease_name}: SHAP Dependence — {top_feature}")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"{_slug(disease_name)}_shap_dependence_{top_feature}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def workflow_diagram(figures_dir):
    steps = [
        "3 Public Clinical\nDatasets\n(CKD, Heart, Diabetes)",
        "Cleaning &\nImputation",
        "Train / Test\nSplit (80/20)",
        "Train 3 Models\n(RF, XGBoost,\nStacked Ensemble)",
        "Evaluate\n(Accuracy, F1,\nROC-AUC)",
        "SHAP\nExplainability",
        "Deployed\nStreamlit Tool",
    ]
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis("off")
    n = len(steps)
    box_w, box_h, gap = 2.0, 1.1, 0.5
    for i, step in enumerate(steps):
        x = i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 0), box_w, box_h, facecolor="#4C72B0", edgecolor="black",
                                    alpha=0.85, zorder=2))
        ax.text(x + box_w / 2, box_h / 2, step, ha="center", va="center", color="white",
                fontsize=9, wrap=True, zorder=3)
        if i < n - 1:
            arrow = mpatches.FancyArrowPatch(
                (x + box_w, box_h / 2), (x + box_w + gap, box_h / 2),
                arrowstyle="-|>", mutation_scale=20, linewidth=2, color="black", zorder=1,
            )
            ax.add_patch(arrow)
    ax.set_xlim(-0.3, n * (box_w + gap))
    ax.set_ylim(-0.3, box_h + 0.3)
    ax.set_title("Unified Framework: End-to-End Methodology", fontsize=13)
    fig.tight_layout()
    out = os.path.join(figures_dir, "workflow_diagram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def save_model(model, disease_name, models_dir):
    out = os.path.join(models_dir, f"{disease_name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(model, out)
    print("saved", out)
    return out
