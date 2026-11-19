"""Re-derives the exact cleaned feature set used to train each disease's
model (same cleaning logic as in the notebook) and writes a schema JSON per
disease: feature name, min, max, median, and whether it's an integer-valued
field. The Streamlit app uses this to build sensible input widgets without
guessing ranges.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
APP_DIR = os.path.join(BASE, "app")
os.makedirs(APP_DIR, exist_ok=True)


def schema_from_df(X):
    schema = []
    for col in X.columns:
        series = X[col]
        is_int = bool((series.dropna() % 1 == 0).all())
        schema.append({
            "name": col,
            "min": float(series.min()),
            "max": float(series.max()),
            "default": float(series.median()),
            "is_int": is_int,
        })
    return schema


def ckd_schema():
    ckd = pd.read_csv(os.path.join(DATA_DIR, "ckd.csv"))
    ckd.columns = [c.strip() for c in ckd.columns]
    ckd["class"] = ckd["class"].astype(str).str.strip().replace({"ckd\t": "ckd"})
    numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    for c in numeric_cols:
        ckd[c] = pd.to_numeric(ckd[c], errors="coerce")
    for c in categorical_cols:
        ckd[c] = ckd[c].astype(str).str.strip().replace({"nan": np.nan})
    for c in numeric_cols:
        ckd[c] = ckd[c].fillna(ckd[c].median())
    cat_options = {}
    for c in categorical_cols:
        ckd[c] = ckd[c].fillna(ckd[c].mode()[0])
        le = LabelEncoder()
        ckd[c] = le.fit_transform(ckd[c])
        cat_options[c] = list(le.classes_)
    X = ckd.drop(columns=["class"])
    return schema_from_df(X), cat_options


def heart_schema():
    heart = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
    for c in heart.columns:
        heart[c] = pd.to_numeric(heart[c], errors="coerce")
    for c in heart.columns:
        heart[c] = heart[c].fillna(heart[c].median())
    X = heart.drop(columns=["target"])
    return schema_from_df(X), {}


def diabetes_schema():
    diabetes = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_as_missing:
        diabetes[c] = diabetes[c].replace(0, np.nan)
        diabetes[c] = diabetes[c].fillna(diabetes[c].median())
    X = diabetes.drop(columns=["Outcome"])
    return schema_from_df(X), {}


if __name__ == "__main__":
    out = {}
    for name, fn in [
        ("Chronic Kidney Disease", ckd_schema),
        ("Heart Disease", heart_schema),
        ("Diabetes", diabetes_schema),
    ]:
        fields, cat_options = fn()
        out[name] = {"fields": fields, "categorical_options": cat_options}
        print(name, "-", len(fields), "fields")

    schema_path = os.path.join(APP_DIR, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", schema_path)
