"""Download the three public clinical datasets used in this research project.

Run once. Saves clean CSVs into the project's data/ folder:
  - ckd.csv        Chronic Kidney Disease (UCI id=336)
  - heart.csv      Heart Disease / Cleveland (UCI id=45)
  - diabetes.csv   Pima Indians Diabetes (public GitHub mirror of the UCI/Kaggle data)
"""
import os
import urllib.request

from ucimlrepo import fetch_ucirepo

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def save_ckd():
    ds = fetch_ucirepo(id=336)
    df = ds.data.features.copy()
    df["class"] = ds.data.targets
    out = os.path.join(DATA_DIR, "ckd.csv")
    df.to_csv(out, index=False)
    print("saved", out, df.shape)


def save_heart():
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets
    out = os.path.join(DATA_DIR, "heart.csv")
    df.to_csv(out, index=False)
    print("saved", out, df.shape)


def save_diabetes():
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    out = os.path.join(DATA_DIR, "diabetes.csv")
    urllib.request.urlretrieve(url, out)
    # add the header row (the raw file has none)
    with open(out, "r") as f:
        rows = f.read()
    with open(out, "w") as f:
        f.write(",".join(columns) + "\n" + rows)
    print("saved", out)


if __name__ == "__main__":
    save_ckd()
    save_heart()
    save_diabetes()
    print("all three datasets downloaded")
