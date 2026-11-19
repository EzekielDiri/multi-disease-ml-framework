"""Multi-Disease Risk Predictor — a small deployed demonstration of the
research notebook's three trained models (Chronic Kidney Disease, Heart
Disease, Diabetes), each with a per-prediction SHAP explanation.

Run locally:   streamlit run app.py
Deploy free:   push this folder's contents to a GitHub repo, then connect
               that repo on share.streamlit.io (Streamlit Community Cloud).
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# --- compatibility fix: same xgboost/shap base_score issue as the notebook ---
import shap.explainers._tree as _shap_tree

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
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Multi-Disease Risk Predictor", layout="centered")

MODEL_FILES = {
    "Chronic Kidney Disease": "chronic_kidney_disease_model.joblib",
    "Heart Disease": "heart_disease_model.joblib",
    "Diabetes": "diabetes_model.joblib",
}


@st.cache_resource
def load_schema():
    with open(os.path.join(HERE, "feature_schema.json")) as f:
        return json.load(f)


@st.cache_resource
def load_model(disease):
    return joblib.load(os.path.join(HERE, MODEL_FILES[disease]))


def build_input_form(disease, schema):
    fields = schema[disease]["fields"]
    cat_options = schema[disease]["categorical_options"]
    values = {}
    cols = st.columns(2)
    for i, field in enumerate(fields):
        name = field["name"]
        col = cols[i % 2]
        with col:
            if name in cat_options:
                choice = st.selectbox(name, cat_options[name], key=f"{disease}_{name}")
                values[name] = cat_options[name].index(choice)
            elif field["is_int"]:
                values[name] = st.number_input(
                    name, min_value=int(field["min"]), max_value=int(field["max"]) * 2 + 1,
                    value=int(field["default"]), step=1, key=f"{disease}_{name}",
                )
            else:
                values[name] = st.number_input(
                    name, min_value=0.0, max_value=float(field["max"]) * 3 + 1,
                    value=float(field["default"]), step=0.1, key=f"{disease}_{name}",
                )
    return values


def explain_prediction(model, X_row, feature_names):
    explainer = shap.TreeExplainer(model) if hasattr(model, "get_booster") or model.__class__.__name__ == "RandomForestClassifier" else None
    if explainer is None:
        return None
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = sv[0]
    order = np.argsort(np.abs(sv))[::-1][:8]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#C44E52" if sv[i] > 0 else "#4C72B0" for i in order]
    ax.barh([feature_names[i] for i in order][::-1], sv[order][::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value (pushes risk up / down)")
    ax.set_title("Why the model said this")
    fig.tight_layout()
    return fig


st.title("Multi-Disease Risk Predictor")
st.caption(
    "A demonstration tool built on the same ensemble models used in the research paper "
    "'A Unified Explainable Ensemble Machine Learning Framework for Early Prediction of "
    "Multiple Chronic Diseases Using Routine Clinical Biomarkers.' For research and "
    "educational demonstration only — not a diagnostic device."
)

schema = load_schema()
disease = st.selectbox("Choose a condition to check", list(MODEL_FILES.keys()))

st.subheader(f"Enter clinical values for {disease}")
values = build_input_form(disease, schema)

if st.button("Predict risk", type="primary"):
    model = load_model(disease)
    feature_order = [f["name"] for f in schema[disease]["fields"]]
    X_row = pd.DataFrame([[values[name] for name in feature_order]], columns=feature_order)

    proba = model.predict_proba(X_row)[0][1]
    pred = int(proba >= 0.5)

    st.markdown("---")
    if pred:
        st.error(f"Predicted risk: **HIGH** ({proba * 100:.1f}% estimated probability)")
    else:
        st.success(f"Predicted risk: **LOW** ({proba * 100:.1f}% estimated probability)")

    with st.spinner("Explaining this prediction..."):
        fig = explain_prediction(model, X_row, feature_order)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("Explanation isn't available for this model type.")

    st.caption(
        "This estimate comes from a machine learning model trained on a public research "
        "dataset. It is not a medical diagnosis — consult a qualified clinician for any "
        "real health concern."
    )
