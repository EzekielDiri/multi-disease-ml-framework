"""Multi-Disease Risk Predictor. A small demonstration of the research
notebook's three trained models (Chronic Kidney Disease, Heart Disease,
Diabetes), each with a per-prediction SHAP explanation.

Run with: streamlit run app.py
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

st.set_page_config(page_title="Multi-Disease Risk Predictor", page_icon="🩺", layout="centered")

MODEL_FILES = {
    "Chronic Kidney Disease": "chronic_kidney_disease_model.joblib",
    "Heart Disease": "heart_disease_model.joblib",
    "Diabetes": "diabetes_model.joblib",
}

# Full, spelled-out labels for every field. The raw dataset columns use lab
# shorthand (sg, bu, sod, rbcc...) that means nothing without a medical
# background, so the form shows the real name and the model keeps the
# original column name underneath.
FEATURE_LABELS = {
    # Chronic Kidney Disease
    "age": "Age (years)",
    "bp": "Blood Pressure (mm Hg)",
    "sg": "Urine Specific Gravity",
    "al": "Albumin Level (0-5 scale)",
    "su": "Sugar Level (0-5 scale)",
    "rbc": "Red Blood Cells in Urine",
    "pc": "Pus Cells in Urine",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria in Urine",
    "bgr": "Random Blood Glucose (mg/dL)",
    "bu": "Blood Urea (mg/dL)",
    "sc": "Serum Creatinine (mg/dL)",
    "sod": "Sodium Level (mEq/L)",
    "pot": "Potassium Level (mEq/L)",
    "hemo": "Hemoglobin (g/dL)",
    "pcv": "Packed Cell Volume (%)",
    "wbcc": "White Blood Cell Count (cells/cumm)",
    "rbcc": "Red Blood Cell Count (millions/cmm)",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Pedal Edema (Leg Swelling)",
    "ane": "Anemia",
    # Heart Disease
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar Above 120 mg/dL",
    "restecg": "Resting ECG Result",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Major Vessels Visible on Fluoroscopy",
    "thal": "Thalassemia Test Result",
    # Diabetes
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Plasma Glucose Concentration (mg/dL)",
    "BloodPressure": "Diastolic Blood Pressure (mm Hg)",
    "SkinThickness": "Triceps Skinfold Thickness (mm)",
    "Insulin": "2-Hour Serum Insulin (mu U/mL)",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function Score",
    "Age": "Age (years)",
}

# Heart Disease stores several fields as numeric codes rather than text
# (the UCI dataset's own convention). Mapped to real options here so nobody
# has to type a bare code they'd have to look up.
HEART_CATEGORY_MAPS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},
}

# Groups the raw fields into labelled sections so a 24-field form like CKD
# reads as a short, organised clinical chart instead of one flat wall of
# inputs. A field missing from every group (schema drift) still renders,
# just folded into a catch-all section at the end.
FIELD_GROUPS = {
    "Chronic Kidney Disease": [
        ("Vitals", "heart-pulse", ["age", "bp"]),
        ("Urine Tests", "droplet", ["sg", "al", "su", "rbc", "pc", "pcc", "ba"]),
        ("Blood Tests", "activity", ["bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]),
        ("Medical History", "clipboard", ["htn", "dm", "cad", "appet", "pe", "ane"]),
    ],
    "Heart Disease": [
        ("Vitals", "heart-pulse", ["age", "sex", "trestbps", "chol"]),
        ("Symptoms", "alert", ["cp", "fbs", "restecg", "thalach", "exang", "oldpeak"]),
        ("Diagnostic Tests", "activity", ["slope", "ca", "thal"]),
    ],
    "Diabetes": [
        ("Personal Info", "heart-pulse", ["Pregnancies", "Age"]),
        ("Test Results", "droplet", ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]),
    ],
}

GROUP_ICONS = {
    "heart-pulse": "\U0001FA7A",
    "droplet": "\U0001F4A7",
    "activity": "\U0001F4CA",
    "clipboard": "\U0001F4CB",
    "alert": "⚠️",
}

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: #EEF2F7;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 800px;}

.app-header {
    padding: 1.75rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
    color: white;
    margin-bottom: 1.1rem;
    box-shadow: 0 12px 28px -8px rgba(30, 64, 175, 0.45);
}
.app-header h1 {color: white; font-size: 1.75rem; margin: 0 0 0.4rem 0;}
.app-header p {color: #DBEAFE; font-size: 0.95rem; margin: 0; line-height: 1.5;}

.stat-row {display: flex; gap: 0.75rem; margin-bottom: 1.75rem;}
.stat-chip {
    flex: 1;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.7rem 0.5rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.stat-chip-num {display: block; font-size: 1.05rem; font-weight: 800; color: #1D4ED8;}
.stat-chip-label {display: block; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: #64748B; margin-top: 0.15rem;}

.condition-picker-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #64748B; margin: 0 0 0.5rem 0.1rem;
}
div[data-testid="column"] div[data-testid="stButton"] button {
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    font-weight: 600;
}
div[data-testid="column"] div[data-testid="stButton"] button[kind="primary"] {
    box-shadow: 0 6px 16px -4px rgba(37, 99, 235, 0.5);
}

div[data-testid="stForm"] {
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.5rem 1.5rem 0.5rem 1.5rem;
    background: #FBFCFE;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-top: 1.5rem;
}
/* Force readable label/text color inside the light card, regardless of
   whether the viewer's Streamlit theme defaults to light or dark text. */
div[data-testid="stForm"] label,
div[data-testid="stForm"] label p,
div[data-testid="stForm"] div[data-testid="stWidgetLabel"] p,
div[data-testid="stForm"] span {
    color: #0F172A !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    color: #0F172A !important;
}

.result-card {
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0 1rem 0;
    border-left: 6px solid;
    box-shadow: 0 8px 20px -6px rgba(15, 23, 42, 0.12);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.result-high {background: #FEF2F2; border-color: #DC2626;}
.result-low {background: #F0FDF4; border-color: #16A34A;}
.result-card .risk-label {font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;}
.result-high .risk-label {color: #B91C1C;}
.result-low .risk-label {color: #15803D;}
.result-card .risk-sub {font-size: 0.9rem; color: #475569;}

.risk-gauge {
    width: 108px; height: 108px; border-radius: 50%; flex-shrink: 0;
    background: conic-gradient(var(--gauge-color) calc(var(--gauge-pct) * 1%), #E2E8F0 0);
    display: flex; align-items: center; justify-content: center;
}
.risk-gauge-inner {
    width: 84px; height: 84px; border-radius: 50%; background: #FFFFFF;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.04);
}
.risk-gauge-value {font-size: 1.35rem; font-weight: 800; color: #0F172A; line-height: 1;}
.risk-gauge-unit {font-size: 0.65rem; font-weight: 700; color: #94A3B8; margin-top: 0.15rem;}

.section-title {font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0 0 1.1rem 0;}
.disclaimer {font-size: 0.8rem; color: #64748B; margin-top: 1rem; line-height: 1.5;}

.field-group-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1D4ED8;
    margin: 1.3rem 0 0.7rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid #DBEAFE;
}
.field-group-title:first-of-type {margin-top: 0;}
.field-group-title .icon {font-size: 0.95rem;}

div[data-testid="stForm"] div[data-testid="stVerticalBlock"] div[data-testid="stNumberInput"],
div[data-testid="stForm"] div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] {
    margin-bottom: 0.4rem;
}
</style>
"""


@st.cache_resource
def load_schema():
    with open(os.path.join(HERE, "feature_schema.json")) as f:
        return json.load(f)


@st.cache_resource
def load_model(disease):
    return joblib.load(os.path.join(HERE, MODEL_FILES[disease]))


def field_label(name):
    return FEATURE_LABELS.get(name, name)


def render_field(disease, field, cat_options, col, values):
    name = field["name"]
    label = field_label(name)
    with col:
        if name in cat_options:
            choice = st.selectbox(label, cat_options[name], key=f"{disease}_{name}")
            values[name] = cat_options[name].index(choice)
        elif disease == "Heart Disease" and name in HEART_CATEGORY_MAPS:
            option_map = HEART_CATEGORY_MAPS[name]
            choice = st.selectbox(label, list(option_map.values()), key=f"{disease}_{name}")
            reverse = {v: k for k, v in option_map.items()}
            values[name] = reverse[choice]
        elif field["is_int"]:
            values[name] = st.number_input(
                label, min_value=int(field["min"]), max_value=int(field["max"]) * 2 + 1,
                value=int(field["default"]), step=1, key=f"{disease}_{name}",
            )
        else:
            values[name] = st.number_input(
                label, min_value=0.0, max_value=float(field["max"]) * 3 + 1,
                value=float(field["default"]), step=0.1, key=f"{disease}_{name}",
            )


def build_input_form(disease, schema):
    fields = schema[disease]["fields"]
    cat_options = schema[disease]["categorical_options"]
    field_by_name = {f["name"]: f for f in fields}
    values = {}

    groups = list(FIELD_GROUPS.get(disease, []))
    grouped_names = {name for _, _, names in groups for name in names}
    leftover = [f["name"] for f in fields if f["name"] not in grouped_names]
    if leftover:
        groups.append(("Other Details", "clipboard", leftover))

    for group_label, icon_key, names in groups:
        present = [n for n in names if n in field_by_name]
        if not present:
            continue
        icon = GROUP_ICONS.get(icon_key, "")
        st.markdown(
            f'<div class="field-group-title"><span class="icon">{icon}</span>{group_label}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, name in enumerate(present):
            render_field(disease, field_by_name[name], cat_options, cols[i % 2], values)

    return values


def explain_prediction(model, X_row, feature_names):
    is_tree_model = hasattr(model, "get_booster") or model.__class__.__name__ == "RandomForestClassifier"
    if not is_tree_model:
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = sv[0]
    order = np.argsort(np.abs(sv))[::-1][:8]
    labels = [field_label(feature_names[i]) for i in order][::-1]
    values = sv[order][::-1]

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in values]
    ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color="#94A3B8", linewidth=0.8)
    ax.set_xlabel("Effect on predicted risk (right = higher, left = lower)")
    ax.set_title("What drove this prediction", fontsize=12, fontweight="bold", loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>🩺 Multi-Disease Risk Predictor</h1>
        <p>A demonstration tool built on the models from the research paper
        "A Unified Explainable Ensemble Machine Learning Framework for Early
        Prediction of Multiple Chronic Diseases Using Routine Clinical
        Biomarkers." For research and educational use only. It is not a
        diagnostic device.</p>
    </div>
    <div class="stat-row">
        <div class="stat-chip"><span class="stat-chip-num">3</span><span class="stat-chip-label">Conditions</span></div>
        <div class="stat-chip"><span class="stat-chip-num">SHAP</span><span class="stat-chip-label">Explainable</span></div>
        <div class="stat-chip"><span class="stat-chip-num">Live</span><span class="stat-chip-label">Model</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

schema = load_schema()

if "disease" not in st.session_state:
    st.session_state.disease = list(MODEL_FILES.keys())[0]

CONDITION_ICONS = {"Chronic Kidney Disease": "🫘", "Heart Disease": "❤️", "Diabetes": "🩸"}

st.markdown('<div class="condition-picker-label">Choose a condition to check</div>', unsafe_allow_html=True)
condition_cols = st.columns(3)
for i, cond in enumerate(MODEL_FILES.keys()):
    with condition_cols[i]:
        is_active = st.session_state.disease == cond
        if st.button(
            f"{CONDITION_ICONS[cond]}  {cond}",
            key=f"condition_{cond}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.disease = cond
            st.rerun()

disease = st.session_state.disease

with st.form(key="prediction_form"):
    st.markdown(f'<div class="section-title">Clinical values for {disease}</div>', unsafe_allow_html=True)
    values = build_input_form(disease, schema)
    submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

if submitted:
    model = load_model(disease)
    feature_order = [f["name"] for f in schema[disease]["fields"]]
    X_row = pd.DataFrame([[values[name] for name in feature_order]], columns=feature_order)

    proba = float(model.predict_proba(X_row)[0][1])
    pred = int(proba >= 0.5)

    css_class = "result-high" if pred else "result-low"
    risk_word = "HIGH" if pred else "LOW"
    gauge_color = "#DC2626" if pred else "#16A34A"
    gauge_pct = max(min(proba * 100, 100.0), 0.0)
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="risk-gauge" style="--gauge-pct: {gauge_pct}; --gauge-color: {gauge_color};">
                <div class="risk-gauge-inner">
                    <span class="risk-gauge-value">{proba * 100:.0f}%</span>
                    <span class="risk-gauge-unit">RISK</span>
                </div>
            </div>
            <div>
                <div class="risk-label">Predicted risk: {risk_word}</div>
                <div class="risk-sub">Estimated probability: {proba * 100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Explaining this prediction..."):
        fig = explain_prediction(model, X_row, feature_order)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("An explanation is not available for this model type.")

    st.markdown(
        """
        <p class="disclaimer">This estimate comes from a machine learning model trained on a
        public research dataset. It is not a medical diagnosis. Talk to a qualified
        clinician about any real health concern.</p>
        """,
        unsafe_allow_html=True,
    )
