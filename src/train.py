"""
Train Logistic Regression and Random Forest on preprocessed arrays;
evaluate and persist models under models/.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _project_root() -> str:
    """Return churn-prediction project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_train_test(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load X/y train and test numpy arrays produced by preprocess.py."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit logistic regression with fixed max_iter and random_state for reproducibility."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Fit random forest with requested hyperparameters."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute accuracy, AUC-ROC, print classification report and confusion matrix;
    return metrics dict for tabular comparison.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {roc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    print(f"AUC (curve integral check): {auc(fpr, tpr):.4f}")
    return {"model": name, "accuracy": acc, "auc_roc": roc}


def print_comparison_table(rows: list[dict]) -> None:
    """Print a simple ASCII comparison of accuracy and AUC across models."""
    print("\n=== Metric comparison ===")
    header = f"{'Model':<22} {'Accuracy':>10} {'AUC-ROC':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<22} {r['accuracy']:>10.4f} {r['auc_roc']:>10.4f}")


def save_models(models_dir: str, lr_model: LogisticRegression, rf_model: RandomForestClassifier) -> None:
    """Persist trained classifiers to pickle files."""
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "logistic_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(models_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)


def main() -> None:
    """Train both models, evaluate, compare, and save pickles."""
    root = _project_root()
    data_dir = os.path.join(root, "data")
    models_dir = os.path.join(root, "models")

    X_train, X_test, y_train, y_test = load_train_test(data_dir)

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    metrics = [
        evaluate_model("Logistic Regression", lr, X_test, y_test),
        evaluate_model("Random Forest", rf, X_test, y_test),
    ]
    print_comparison_table(metrics)

    min_auc = min(m["auc_roc"] for m in metrics)
    if min_auc < 0.85:
        print(f"\nWarning: minimum AUC-ROC is {min_auc:.4f} (target >= 0.85).")
    else:
        print("\nAll models meet AUC-ROC >= 0.85 on the holdout set.")

    save_models(models_dir, lr, rf)
    print(f"\nSaved models to {models_dir}/")


if __name__ == "__main__":
    main()
