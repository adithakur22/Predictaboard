"""
insights.py — Insights tab: feature importance, correlation,
categorical breakdowns, and numeric vs target box plots.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from Utils import COLORS, encode_target


def render_insights(
    df: pd.DataFrame,
    target_col: str,
    models: dict,
    feature_names: list,
    model_name: str,
) -> None:
    active_model = models.get(model_name)
    imp = getattr(active_model, "feature_importances_", None)

    # ── Feature importance (RF and XGBoost only) ─────────────────────────────
    if imp is not None and len(imp) == len(feature_names):
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        fig_fi = px.bar(
            fi, x="importance", y="feature", orientation="h",
            title=f"{model_name} — top 20 feature importances",
            color="importance", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info(f"Feature importances not available for {model_name}.")

    if target_col not in df.columns:
        st.info("Select a target column to unlock insight charts.")
        return

    y = encode_target(df[target_col])

    # ── Numeric correlates with target ───────────────────────────────────────
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
    if num_cols:
        corr = (
            df[num_cols].corrwith(y).abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        corr.columns = ["feature", "|correlation|"]
        fig_corr = px.bar(
            corr, x="|correlation|", y="feature", orientation="h",
            title=f"Top numeric features correlated with `{target_col}`",
            color="|correlation|", color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Categorical breakdown ────────────────────────────────────────────────
    cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns if c != target_col]
    if cat_cols:
        chosen_cat = st.selectbox("Categorical column to break down", cat_cols, key="ins_cat")
        tmp = df[[chosen_cat]].copy()
        tmp["_y"] = y.values
        grp = (
            tmp.groupby(chosen_cat, as_index=False)["_y"]
            .mean()
            .rename(columns={"_y": "positive_rate"})
            .sort_values("positive_rate", ascending=False)
        )
        fig_cat = px.bar(
            grp, x=chosen_cat, y="positive_rate",
            title=f"Positive rate by `{chosen_cat}`",
            color="positive_rate", color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Numeric vs target box plot ───────────────────────────────────────────
    if num_cols:
        chosen_num = st.selectbox(
            "Numeric column to compare across classes", num_cols, key="ins_num"
        )
        tmp2 = df[[chosen_num]].copy()
        tmp2["_y"] = y.map({0: "Negative (0)", 1: "Positive (1)"}).values
        fig_box = px.box(
            tmp2, x="_y", y=chosen_num,
            title=f"`{chosen_num}` by target class",
            color="_y",
            color_discrete_sequence=[COLORS["success"], COLORS["danger"]],
        )
        st.plotly_chart(fig_box, use_container_width=True)