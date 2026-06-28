"""
utils.py — Shared constants, helpers, and validators used across all modules.
"""
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────
RISK_BINS   = [0.0, 0.33, 0.66, 1.01]
RISK_LABELS = ["Low", "Medium", "High"]
COLORS      = {
    "primary": "#636EFA",
    "danger":  "#EF553B",
    "success": "#00CC96",
    "warn":    "#FFA15A",
}


# ── Target encoding ──────────────────────────────────────────────────────────
def encode_target(series: pd.Series) -> pd.Series:
    """Convert any binary column to 0/1 integers."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        unique = series.dropna().unique()
        if set(unique).issubset({0, 1}):
            return series.astype(int)
        median = series.median()
        return (series > median).astype(int)
    le = LabelEncoder()
    return pd.Series(le.fit_transform(series.astype(str)), index=series.index)


# ── Target validation ────────────────────────────────────────────────────────
def validate_target(series: pd.Series, col_name: str) -> bool:
    """Return False and show st.error if column is not a valid binary target."""
    unique_vals = series.dropna().unique()
    if len(unique_vals) < 2:
        st.error(f"Column `{col_name}` has only 1 unique value — not a valid target.")
        return False
    if len(unique_vals) > 10:
        st.error(
            f"Column `{col_name}` has {len(unique_vals)} unique values — looks like a "
            f"continuous or ID column. Pick a binary column (2 classes like Yes/No or 0/1)."
        )
        return False
    if len(unique_vals) > 2:
        st.warning(
            f"Column `{col_name}` has {len(unique_vals)} unique values. "
            f"Only binary targets are fully supported — results may vary."
        )
    return True


# ── ID column detection ──────────────────────────────────────────────────────
def auto_drop_id_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Drop columns where every value is unique (ID-like). Returns (df, dropped_names)."""
    id_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    return df.drop(columns=id_cols), id_cols


# ── Safe transform ───────────────────────────────────────────────────────────
def safe_transform(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """Transform only the columns the preprocessor was fitted on."""
    fitted_cols = []
    for _, _, cols in preprocessor.transformers_:
        fitted_cols.extend(cols if isinstance(cols, list) else [cols])
    df2 = df.reindex(columns=fitted_cols, fill_value=0)
    return preprocessor.transform(df2)


# ── Risk bucketing ───────────────────────────────────────────────────────────
def risk_bucket(proba: np.ndarray) -> pd.Series:
    return pd.cut(proba, bins=RISK_BINS, labels=RISK_LABELS, right=False)


# ── CSV download helper ──────────────────────────────────────────────────────
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ── File hash for cache invalidation ─────────────────────────────────────────
def file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h