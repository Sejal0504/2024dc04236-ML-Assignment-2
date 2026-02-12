import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, ConfusionMatrixDisplay, classification_report
)

from xgboost import XGBClassifier


st.set_page_config(page_title="Adult Income Classification", layout="wide")

TARGET = "income"


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", pd.NA).dropna()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def to_binary_y(y: pd.Series) -> np.ndarray:
    y = y.astype(str).str.strip().str.replace(".", "", regex=False)
    return (y == ">50K").astype(int).to_numpy()


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


@st.cache_resource
def train_all_models(train_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET])
    y_train = to_binary_y(train_df[TARGET])

    preprocess = build_preprocess(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    }

    trained = {}
    for name, clf in models.items():
        if name == "Naive Bayes":
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("to_dense", FunctionTransformer(to_dense)),
                ("model", clf)
            ])
        else:
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", clf)
            ])
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    return trained


st.title("Adult Income Prediction (<=50K vs >50K)")
st.caption("Upload a CSV (with 'income') → choose model → view metrics + confusion matrix + report.")

uploaded = st.file_uploader("Upload CSV (test data recommended)", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)

if TARGET not in df.columns:
    st.error("Uploaded CSV must contain 'income' column. Upload your test.csv (with income).")
    st.stop()

df = clean_df(df)

# Split inside app (cloud-friendly)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])

trained_models = train_all_models(train_df)

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(trained_models.keys()))
pipe = trained_models[model_name]

X_test = test_df.drop(columns=[TARGET])
y_true = to_binary_y(test_df[TARGET])

y_pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
c2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
c3.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
c3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")
if proba is not None:
    c1.metric("AUC", f"{roc_auc_score(y_true, proba):.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
st.pyplot(fig, clear_figure=True)

st.subheader("Classification Report")
st.text(classification_report(y_true, y_pred, zero_division=0))

st.subheader("Prediction Preview")
out = test_df.copy()
out["pred_income_gt_50k"] = y_pred
if proba is not None:
    out["proba_gt_50k"] = proba
st.dataframe(out.head(20), use_container_width=True)
