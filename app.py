"""
PredictaBoard — Universal Binary Classification Dashboard
app.py is the entry point only. All logic lives in src/.
"""
import json
import os
import sys
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Allow imports from src/ ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from Charts         import render_data_preview, render_overview        # noqa: E402
from Evaluation     import render_evaluation                            # noqa: E402
from Explainability import render_shap                                  # noqa: E402
from Insights       import render_insights                              # noqa: E402
from Pipeline       import build_dynamic_pipeline                       # noqa: E402
from Predictions    import render_predictions                           # noqa: E402
from Utils          import auto_drop_id_cols, file_hash, validate_target  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Default dataset (telecom fallback)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def _default_dataset() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telecom_churn.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="PredictaBoard", layout="wide", page_icon="📊")
    st.title("📊 PredictaBoard")
    st.caption("Universal binary classification dashboard — works with **any** CSV.")

    # ── Sidebar: file upload ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload any CSV", type=["csv"])

    # ── Load data ────────────────────────────────────────────────────────────
    if uploaded is not None:
        fhash  = file_hash(uploaded)
        df_raw = pd.read_csv(uploaded)
        st.sidebar.success(
            f"**{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols"
        )
    else:
        fhash  = "default"
        df_raw = _default_dataset()
        if df_raw.empty:
            st.error("No default dataset found. Please upload a CSV.")
            st.stop()
        st.sidebar.info("Using bundled telecom_churn.csv")

    all_cols = df_raw.columns.tolist()

    # ── Sidebar: target + controls ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        default_idx = next(
            (i for i, c in enumerate(all_cols)
             if c.lower() in ("churn", "target", "label", "y", "outcome", "attrition")), 0
        )
        target_col = st.selectbox(
            "Target (label) column", all_cols, index=default_idx,
            help="Pick the binary column you want to predict.",
        )
        if not validate_target(df_raw[target_col], target_col):
            st.stop()

        drop_cols = st.multiselect(
            "Drop columns before training",
            [c for c in all_cols if c != target_col],
            help="Exclude ID columns, names, or dates.",
        )

        st.markdown("---")
        st.subheader("Model settings")
        model_name = st.selectbox("Active model", ["Random Forest", "Logistic Regression", "XGBoost"])
        n_estimators = st.slider("RF: number of trees",        50, 500, 200, 50)
        max_depth    = st.slider("RF: max depth (0=unlimited)", 0,  30,   0,  1)

    # ── Build pipeline ───────────────────────────────────────────────────────
    with st.spinner("Fitting preprocessor and training models…"):
        try:
            (preprocessor, models, feature_names,
             X_test_t, y_test, X_test_raw,
             cv_scores, imbalanced, auto_dropped) = build_dynamic_pipeline(
                df_raw.to_json(orient="split"),
                target_col,
                tuple(sorted(drop_cols)),
                n_estimators,
                max_depth,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    st.sidebar.success("✅ Models ready.")
    if imbalanced:
        st.sidebar.warning("Imbalanced target — using balanced class weights.")

    # Clean working dataframe
    df = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], errors="ignore")
    df, _ = auto_drop_id_cols(df)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Data Preview",
        "🗂 Overview",
        "🔮 Predictions",
        "📈 Evaluation",
        "💡 Insights",
        "🧠 Explainability (SHAP)",
    ])

    with tabs[0]:
        render_data_preview(df, target_col, auto_dropped)
    with tabs[1]:
        render_overview(df, target_col, imbalanced)
    with tabs[2]:
        st.caption(f"Scoring with **{model_name}** · target: `{target_col}`")
        render_predictions(df, target_col, model_name, preprocessor, models)
    with tabs[3]:
        render_evaluation(df, target_col, preprocessor, models, X_test_t, y_test, cv_scores)
    with tabs[4]:
       render_insights(df, target_col, models, feature_names, model_name)
    with tabs[5]:
        st.caption("SHAP explains *why* the model made each prediction.")
        render_shap(X_test_t, models, feature_names, model_name)


if __name__ == "__main__":
    main()