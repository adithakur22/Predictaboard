"""
explainability.py — SHAP explainability tab.
TreeExplainer for Random Forest and XGBoost, LinearExplainer for Logistic Regression.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from Utils import to_csv_bytes

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def render_shap(
    X_test_t: np.ndarray,
    models: dict,
    feature_names: list,
    model_name: str,
) -> None:
    if not _HAS_SHAP:
        st.error("SHAP not installed. Run `pip install shap` and restart the app.")
        return

    model = models[model_name]
    st.info(f"Computing SHAP values for **{model_name}** on test split (up to 200 rows).")

    n  = min(200, X_test_t.shape[0])
    Xs = X_test_t[:n]

    # ── Compute SHAP values ──────────────────────────────────────────────────
    try:
        if model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            if len(sv.shape) > 2:
                sv = sv.reshape(sv.shape[0], -1)

        elif model_name == "XGBoost":
            # Must pass the underlying booster — not the sklearn wrapper
            explainer = shap.TreeExplainer(model.get_booster())
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]

        else:  # Logistic Regression
            explainer = shap.LinearExplainer(model, Xs)
            sv = explainer.shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) else sv

    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return

    # ── Align feature names to actual SHAP output width ─────────────────────
    n_shap_cols = sv.shape[1] if sv.ndim > 1 else sv.shape[0]
    if len(feature_names) == n_shap_cols:
        actual_names = feature_names
    elif n_shap_cols < len(feature_names):
        actual_names = feature_names[:n_shap_cols]
    else:
        actual_names = list(feature_names) + [
            f"feature_{i}" for i in range(len(feature_names), n_shap_cols)
        ]

    mean_abs = np.abs(sv).mean(axis=0).flatten()

    if len(actual_names) != len(mean_abs):
        st.error(
            f"Feature name / SHAP value mismatch: "
            f"{len(actual_names)} names vs {len(mean_abs)} SHAP values."
        )
        return

    # ── Mean absolute SHAP bar chart ─────────────────────────────────────────
    shap_df = (
        pd.DataFrame({"feature": actual_names, "mean_|SHAP|": mean_abs})
        .sort_values("mean_|SHAP|", ascending=False)
        .head(20)
    )
    fig_shap = px.bar(
        shap_df, x="mean_|SHAP|", y="feature", orientation="h",
        title=f"SHAP — mean absolute impact ({model_name})",
        color="mean_|SHAP|", color_continuous_scale="Purples",
    )
    fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Beeswarm scatter ─────────────────────────────────────────────────────
    with st.expander("SHAP beeswarm — top 10 features"):
        top_feats = shap_df["feature"].head(10).tolist()
        top_idx   = [
            actual_names.index(f)
            for f in top_feats
            if f in actual_names and actual_names.index(f) < Xs.shape[1]
        ]
        top_feats = [actual_names[i] for i in top_idx]

        sv_top = sv[:, top_idx]
        fv_top = Xs[:, top_idx]

        rows = [
            {
                "feature":       top_feats[fi],
                "SHAP value":    float(sv_top[si, fi]),
                "Feature value": float(fv_top[si, fi]),
            }
            for fi in range(len(top_feats))
            for si in range(sv_top.shape[0])
        ]
        bee_df  = pd.DataFrame(rows)
        fig_bee = px.scatter(
            bee_df, x="SHAP value", y="feature", color="Feature value",
            color_continuous_scale="RdBu",
            title="SHAP beeswarm — top 10 features",
            opacity=0.7,
        )
        fig_bee.update_traces(marker=dict(size=6))
        fig_bee.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bee, use_container_width=True)

        st.download_button(
            "⬇ Download SHAP values as CSV",
            data=to_csv_bytes(pd.DataFrame(sv, columns=actual_names)),
            file_name="shap_values.csv",
            mime="text/csv",
        )