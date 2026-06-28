"""
Load preprocessor + trained model and produce churn probabilities and labels for a feature dataframe.
"""

from __future__ import annotations

import os
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def _project_root() -> str:
    """Return churn-prediction project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_preprocessor(models_dir: str):
    """Load fitted ColumnTransformer saved by preprocess.py."""
    path = os.path.join(models_dir, "preprocessor.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(models_dir: str, model_name: Literal["logistic", "rf"]) -> BaseEstimator:
    """Load either logistic regression or random forest pickle."""
    fname = "logistic_model.pkl" if model_name == "logistic" else "rf_model.pkl"
    path = os.path.join(models_dir, fname)
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-feature columns if present; keep the same feature columns as training CSV
    (excluding customer_id and churn).
    """
    drop_cols = [c for c in ("customer_id", "churn") if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    expected = ["tenure_months", "monthly_charges", "total_charges", "contract_type", "internet_service", "num_support_tickets", "payment_method"]
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return X[expected]


def predict_batch(
    df: pd.DataFrame, preprocessor, model: BaseEstimator
) -> pd.DataFrame:
    """
    Transform rows with the fitted preprocessor and return customer_id, churn_probability, predicted_churn.
    """
    X_df = prepare_features(df)
    X = preprocessor.transform(X_df)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"].values if "customer_id" in df.columns else np.arange(len(df)),
            "churn_probability": proba,
            "predicted_churn": pred.astype(int),
        }
    )
    return out


def main() -> None:
    """Smoke test: run predictions on the raw CSV using Random Forest."""
    root = _project_root()
    csv_path = os.path.join(root, "data", "telecom_churn.csv")
    models_dir = os.path.join(root, "models")

    df = pd.read_csv(csv_path).head(20)
    pre = load_preprocessor(models_dir)
    model = load_model(models_dir, "rf")
    result = predict_batch(df, pre, model)
    print(result.head())


if __name__ == "__main__":
    main()
