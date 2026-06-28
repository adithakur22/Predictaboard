"""
pipeline.py — Dynamic preprocessor fitting and model training.
Works with any CSV by auto-detecting column types.
Models: Random Forest, Logistic Regression, XGBoost.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from Utils import auto_drop_id_cols, encode_target


@st.cache_data(show_spinner=False)
def build_dynamic_pipeline(
    df_json: str,
    target_col: str,
    drop_cols: tuple,
    n_estimators: int,
    max_depth: int,
):
    """
    Fit a ColumnTransformer and train Random Forest, Logistic Regression,
    and XGBoost on any dataframe. Cached per (data hash, target, settings).

    Returns:
        preprocessor, models, feature_names,
        X_test_t, y_test, X_test_raw,
        cv_scores, imbalanced, auto_dropped
    """
    df = pd.read_json(io.StringIO(df_json), orient="split")

    # ── Drop ID-like and user-selected columns ───────────────────────────────
    df, auto_dropped = auto_drop_id_cols(df)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ── Encode target ────────────────────────────────────────────────────────
    y = encode_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Drop constant / all-null columns
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique() > 1]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Detect class imbalance ───────────────────────────────────────────────
    pos_rate   = y.mean()
    imbalanced = bool(pos_rate < 0.2 or pos_rate > 0.8)
    cw = "balanced" if imbalanced else None

    # XGBoost uses scale_pos_weight instead of class_weight
    # scale_pos_weight = negative samples / positive samples
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    scale_pos = round(neg_count / pos_count, 2) if imbalanced and pos_count > 0 else 1

    # ── Build ColumnTransformer ──────────────────────────────────────────────
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── Expanded feature names (post-OHE) ───────────────────────────────────
    feat_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feat_names.extend(cols)
        elif name == "cat":
            feat_names.extend(transformer.get_feature_names_out(cols).tolist())

    # ── Train models ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_t, y_train)

    lr = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
    lr.fit(X_train_t, y_train)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else 6,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train_t, y_train)

    models = {
        "Random Forest":       rf,
        "Logistic Regression": lr,
        "XGBoost":             xgb,
    }

    # ── 5-fold cross-validation AUC ─────────────────────────────────────────
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_full_t = preprocessor.transform(X)
    cv_scores = {
        name: cross_val_score(model, X_full_t, y, cv=cv, scoring="roc_auc")
        for name, model in models.items()
    }

    return (
        preprocessor, models, feat_names,
        X_test_t, y_test.values, X_test,
        cv_scores, imbalanced, auto_dropped,
    )