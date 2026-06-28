"""
Load raw CSV, impute missing values, one-hot encode categoricals,
scale numerics with StandardScaler, train/test split, save arrays and artifacts.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _project_root() -> str:
    """Return churn-prediction project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer: median imputation + scaling for numerics;
    most_frequent imputation + one-hot for categoricals (unknown categories ignored at transform).
    """
    numeric_features = ["tenure_months", "monthly_charges", "total_charges", "num_support_tickets"]
    categorical_features = ["contract_type", "internet_service", "payment_method"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def load_and_transform(
    csv_path: str, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Load CSV, fit preprocessor on train split, transform train/test, return arrays and feature names.
    """
    df = pd.read_csv(csv_path)
    if "churn" not in df.columns:
        raise ValueError("Expected column 'churn' in dataset.")

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"].astype(int).to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    feature_names = list(preprocessor.get_feature_names_out())

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def save_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    data_dir: str,
    models_dir: str,
) -> None:
    """Write numpy arrays, scaler-only pickle (numeric StandardScaler), preprocessor bundle, and feature names JSON."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    # User-requested scaler.pkl: persist the StandardScaler from the numeric branch only
    num_pipe: Pipeline = preprocessor.named_transformers_["num"]
    scaler: StandardScaler = num_pipe.named_steps["scaler"]
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(data_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)


def main() -> None:
    """End-to-end preprocessing with paths relative to project root."""
    root = _project_root()
    csv_path = os.path.join(root, "data", "telecom_churn.csv")
    data_dir = os.path.join(root, "data")
    models_dir = os.path.join(root, "models")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run data/generate_data.py first.")

    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_transform(csv_path, random_state=42)
    save_artifacts(X_train, X_test, y_train, y_test, preprocessor, feature_names, data_dir, models_dir)

    print(f"Saved arrays to {data_dir}/")
    print(f"Saved scaler.pkl and preprocessor.pkl to {models_dir}/")
    print(f"Feature dimension: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
