import os, json
import numpy as np
import pandas as pd
import joblib

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

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
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

from xgboost import XGBClassifier

TRAIN_PATH = r"D:\DES\SEM-2\ML\data\processed\train.csv"
TEST_PATH  = r"D:\DES\SEM-2\ML\data\processed\test.csv"
MODEL_DIR  = r"D:\DES\SEM-2\ML\model"
METRICS_JSON = os.path.join(MODEL_DIR, "metrics.json")

TARGET = "income"

def to_binary(y: pd.Series) -> np.ndarray:
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

def get_score(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return pipe.predict(X)

def metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = to_binary(train_df[TARGET])

    X_test = test_df.drop(columns=[TARGET])
    y_test = to_binary(test_df[TARGET])

    preprocess = build_preprocess(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400,
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

    out = {}

    for name, clf in models.items():
        # GaussianNB needs dense input after one-hot
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
        y_pred = pipe.predict(X_test)
        y_score = get_score(pipe, X_test)

        out[name] = metrics(y_test, y_pred, y_score)

        safe = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{safe}.pkl"))
        print(f"Saved: {name} -> model/{safe}.pkl")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved: model/metrics.json")

    # Print comparison table for README/PDF
    rows = []
    for name, m in out.items():
        rows.append([name, m["accuracy"], m["auc"], m["precision"], m["recall"], m["f1"], m["mcc"]])

    table = pd.DataFrame(rows, columns=["ML Model Name","Accuracy","AUC","Precision","Recall","F1","MCC"])
    print("\n=== Comparison Table (copy to README + PDF) ===")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
