"""
evaluation.py — Evaluation tab: confusion matrix, ROC curve,
Precision-Recall curve, CV scores, and model comparison.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from Utils import COLORS


def render_evaluation(
    df: pd.DataFrame,
    target_col: str,
    preprocessor,
    models: dict,
    X_test_t: np.ndarray,
    y_test: np.ndarray,
    cv_scores: dict,
) -> None:
    if target_col not in df.columns:
        st.info("Select a valid target column to see evaluation metrics.")
        return

    # ── Cross-validation scores ──────────────────────────────────────────────
    st.subheader("5-Fold Cross-Validation AUC")
    cv_rows = [
        {
            "Model":    name,
            "Mean AUC": round(scores.mean(), 4),
            "Std":      round(scores.std(),  4),
            "Min":      round(scores.min(),  4),
            "Max":      round(scores.max(),  4),
        }
        for name, scores in cv_scores.items()
    ]
    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    fig_cv = go.Figure()
    for name, scores in cv_scores.items():
        fig_cv.add_trace(go.Box(y=scores, name=name, boxpoints="all", jitter=0.3))
    fig_cv.update_layout(title="CV AUC distribution across 5 folds", yaxis_title="AUC")
    st.plotly_chart(fig_cv, use_container_width=True)

    # ── Per-model deep evaluation ────────────────────────────────────────────
    st.subheader("Model-by-model evaluation")
    model_tabs = st.tabs(list(models.keys()))

    for tab, (mname, model) in zip(model_tabs, models.items()):
        with tab:
            _render_single_model(mname, model, X_test_t, y_test)

    # ── Side-by-side comparison ──────────────────────────────────────────────
    st.subheader("Model comparison — RF vs Logistic Regression")
    comparison_rows = []
    for mname, model in models.items():
        yp    = model.predict_proba(X_test_t)[:, 1]
        ypred = (yp >= 0.5).astype(int)
        rep   = classification_report(y_test, ypred, output_dict=True, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yp)
        except Exception:
            auc = float("nan")
        comparison_rows.append({
            "Model":       mname,
            "AUC-ROC":     round(auc, 4),
            "Precision":   round(rep["1"]["precision"], 4),
            "Recall":      round(rep["1"]["recall"],    4),
            "F1":          round(rep["1"]["f1-score"],  4),
            "CV AUC mean": round(cv_scores[mname].mean(), 4),
            "CV AUC std":  round(cv_scores[mname].std(),  4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig_comp = px.bar(
        comp_df.melt(id_vars="Model", value_vars=["AUC-ROC", "Precision", "Recall", "F1"]),
        x="variable", y="value", color="Model", barmode="group",
        title="RF vs Logistic Regression — metric comparison",
        color_discrete_sequence=[COLORS["primary"], COLORS["danger"]],
    )
    fig_comp.update_layout(yaxis_range=[0, 1], yaxis_title="Score")
    st.plotly_chart(fig_comp, use_container_width=True)


def _render_single_model(
    mname: str,
    model,
    X_test_t: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Render metrics, confusion matrix, ROC, and PR curve for one model."""
    y_prob = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",   f"{auc:.4f}")
    m2.metric("Precision", f"{report['1']['precision']:.4f}")
    m3.metric("Recall",    f"{report['1']['recall']:.4f}")
    m4.metric("F1-score",  f"{report['1']['f1-score']:.4f}")

    col_cm, col_roc = st.columns(2)

    # Confusion matrix
    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Neg (0)", "Pos (1)"],
            y=["Neg (0)", "Pos (1)"],
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    with col_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {auc:.4f}",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline",
            line=dict(dash="dash", color="grey"),
        ))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines",
        line=dict(color=COLORS["warn"], width=2),
    ))
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # Full report
    with st.expander("Full classification report"):
        rep_df = pd.DataFrame(report).T.round(4)
        st.dataframe(rep_df, use_container_width=True)