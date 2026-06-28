"""
predictions.py — Predictions tab renderer.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score

from Utils import COLORS, encode_target, risk_bucket, safe_transform, to_csv_bytes


def render_predictions(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    preprocessor,
    models: dict,
) -> None:
    model = models[model_name]
    X     = df.drop(columns=[target_col], errors="ignore")
    Xt    = safe_transform(X, preprocessor)
    proba = model.predict_proba(Xt)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    results = df.copy()
    results["churn_probability"] = proba.round(4)
    results["predicted_label"]   = pred
    results["risk_bucket"]       = risk_bucket(proba).astype(str)

    # ── Risk bucket summary ──────────────────────────────────────────────────
    bucket_counts = results["risk_bucket"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Low risk",    int(bucket_counts.get("Low",    0)))
    c2.metric("🟡 Medium risk", int(bucket_counts.get("Medium", 0)))
    c3.metric("🔴 High risk",   int(bucket_counts.get("High",   0)))

    # ── Probability histogram ────────────────────────────────────────────────
    fig_hist = px.histogram(
        results, x="churn_probability", nbins=40,
        title="Prediction probability distribution",
        color_discrete_sequence=[COLORS["danger"]],
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Threshold slider ─────────────────────────────────────────────────────
    threshold = st.slider(
        "High-risk threshold", 0.0, 1.0, 0.5, 0.05,
        help="Rows above this probability are flagged as high risk",
    )
    high_risk = results[results["churn_probability"] >= threshold]
    st.metric(f"Customers at or above {threshold:.0%} probability", len(high_risk))

    # ── Full prediction table ────────────────────────────────────────────────
    st.subheader("Full prediction table")
    st.dataframe(results, use_container_width=True)

    # ── Download button ──────────────────────────────────────────────────────
    st.download_button(
        label="⬇ Download predictions as CSV",
        data=to_csv_bytes(results),
        file_name="predictions.csv",
        mime="text/csv",
    )

    # ── AUC if ground truth available ────────────────────────────────────────
    if target_col in df.columns:
        try:
            y_true = encode_target(df[target_col])
            auc    = roc_auc_score(y_true, proba)
            st.metric(f"AUC-ROC vs `{target_col}`", f"{auc:.4f}")
        except ValueError as e:
            st.warning(f"Could not compute AUC: {e}")