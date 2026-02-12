import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

st.set_page_config(page_title="Adult Income Classification", layout="wide")

MODEL_DIR = Path("model")
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET = "income"

# ---- Helpers ----
def pretty(stem: str) -> str:
    return stem.replace("_", " ").title()

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

# ---- UI ----
st.title("Adult Income Prediction (<=50K vs >50K)")
st.caption("Upload the test CSV (recommended) and choose a model. The app shows stored evaluation metrics and can compute a confusion matrix for uploaded data.")

# Model selection (dropdown required)
model_files = sorted([p for p in MODEL_DIR.glob("*.pkl")])
if not model_files:
    st.error("No .pkl models found in /model. Run scripts/train_models.py first.")
    st.stop()

model_map = {pretty(p.stem): p for p in model_files}
chosen_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))
model_path = model_map[chosen_name]

# Dataset upload (required)
uploaded = st.file_uploader("Upload CSV (test data)", type=["csv"])

# Show evaluation metrics (required)
st.subheader("Evaluation Metrics (from saved metrics.json)")
metrics_data = load_metrics()

# metrics.json uses keys like "XGBoost", "kNN", etc.
# map display name back to that
key_guess = None
for k in metrics_data.keys():
    if pretty(k.replace(" ", "_").lower()) == chosen_name:
        key_guess = k
        break
# fallback: try direct matches
if key_guess is None and chosen_name in metrics_data:
    key_guess = chosen_name

if key_guess and key_guess in metrics_data:
    m = metrics_data[key_guess]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.4f}")
    c1.metric("AUC", f"{m['auc']:.4f}")
    c2.metric("Precision", f"{m['precision']:.4f}")
    c2.metric("Recall", f"{m['recall']:.4f}")
    c3.metric("F1", f"{m['f1']:.4f}")
    c3.metric("MCC", f"{m['mcc']:.4f}")

    with st.expander("Saved Classification Report"):
        st.text(m["classification_report"])
else:
    st.info("metrics.json not found (or model key mismatch). You can still predict with the uploaded CSV.")

st.divider()

# ---- Predict on uploaded data ----
st.subheader("Predict on Uploaded Data")

pipe = joblib.load(model_path)

if uploaded is None:
    st.warning("Upload a CSV file to run predictions.")
    st.stop()

df = pd.read_csv(uploaded)

# If income exists, we can compute confusion matrix/report for uploaded data
has_target = TARGET in df.columns
X = df.drop(columns=[TARGET]) if has_target else df.copy()

y_pred = pipe.predict(X)

proba = None
if hasattr(pipe, "predict_proba"):
    proba = pipe.predict_proba(X)[:, 1]

out = df.copy()
out["pred_income_gt_50k"] = y_pred
if proba is not None:
    out["proba_gt_50k"] = proba

st.write("Predictions preview:")
st.dataframe(out.head(20), use_container_width=True)

# Confusion matrix OR classification report (required)
if has_target:
    st.subheader("Confusion Matrix (Uploaded Data)")
    y_true = df[TARGET].astype(str).str.strip().str.replace(".", "", regex=False)
    y_true_bin = (y_true == ">50K").astype(int)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true_bin, y_pred, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Classification Report (Uploaded Data)")
    st.text(classification_report(y_true_bin, y_pred, zero_division=0))

    if proba is not None:
        st.subheader("AUC on Uploaded Data")
        st.write(f"{roc_auc_score(y_true_bin, proba):.4f}")
else:
    st.info("To show confusion matrix/report, upload a CSV that includes the 'income' column (use your data/processed/test.csv).")
