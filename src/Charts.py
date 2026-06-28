"""
charts.py — Data Preview and Overview tab renderers.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from Utils import COLORS, encode_target


# ── Tab 1: Data Preview ───────────────────────────────────────────────────────
def render_data_preview(
    df: pd.DataFrame,
    target_col: str,
    auto_dropped: list,
) -> None:
    st.subheader("Raw data (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",             f"{len(df):,}")
    c2.metric("Columns",          df.shape[1])
    c3.metric("Numeric cols",     df.select_dtypes(include=np.number).shape[1])
    c4.metric("Categorical cols", df.select_dtypes(exclude=np.number).shape[1])

    if auto_dropped:
        st.info(
            f"Auto-dropped ID-like columns (all values unique): "
            f"`{'`, `'.join(auto_dropped)}`"
        )

    # ── Missing value bar chart ──────────────────────────────────────────────
    st.subheader("Missing value map")
    miss = df.isnull().mean().reset_index()
    miss.columns = ["column", "missing_rate"]
    miss = miss[miss["missing_rate"] > 0].sort_values("missing_rate", ascending=False)

    if miss.empty:
        st.success("No missing values found in this dataset.")
    else:
        fig = px.bar(
            miss, x="missing_rate", y="column", orientation="h",
            title="Missing value rate per column",
            color="missing_rate", color_continuous_scale="Reds",
            labels={"missing_rate": "Missing %"},
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Column types summary ─────────────────────────────────────────────────
    with st.expander("Column types and unique counts"):
        dtype_df = pd.DataFrame({
            "dtype":   df.dtypes.astype(str),
            "nunique": df.nunique(),
            "nulls":   df.isnull().sum(),
            "null_%":  (df.isnull().mean() * 100).round(2),
        }).reset_index().rename(columns={"index": "column"})
        st.dataframe(dtype_df, use_container_width=True)


# ── Tab 2: Overview ───────────────────────────────────────────────────────────
def render_overview(
    df: pd.DataFrame,
    target_col: str,
    imbalanced: bool,
) -> None:
    if target_col in df.columns:
        y        = encode_target(df[target_col])
        pos_rate = float(y.mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive rate",    f"{pos_rate:.1%}")
        c2.metric("Positive class (1)", f"{int(y.sum()):,}")
        c3.metric("Negative class (0)", f"{int((1 - y).sum()):,}")

        if imbalanced:
            st.warning(
                "Class imbalance detected — "
                "models trained with `class_weight='balanced'` to compensate."
            )

        counts = y.value_counts().sort_index().reset_index()
        counts.columns = [target_col, "count"]
        counts[target_col] = counts[target_col].map(
            {0: "Negative (0)", 1: "Positive (1)"}
        )
        fig_bar = px.bar(
            counts, x=target_col, y="count",
            title=f"Class distribution — `{target_col}`",
            color=target_col,
            color_discrete_sequence=[COLORS["success"], COLORS["danger"]],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Numeric distributions ────────────────────────────────────────────────
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[target_col] if target_col in df.columns else [],
        errors="ignore",
    )
    if not num_df.empty:
        with st.expander("Numeric distributions", expanded=True):
            cols_per_row = 3
            cols_list    = num_df.columns.tolist()
            for i in range(0, len(cols_list), cols_per_row):
                row = st.columns(cols_per_row)
                for j, col in enumerate(cols_list[i: i + cols_per_row]):
                    with row[j]:
                        fig = px.histogram(
                            df, x=col, nbins=30, title=col,
                            color_discrete_sequence=[COLORS["primary"]],
                        )
                        fig.update_layout(
                            showlegend=False,
                            margin=dict(t=30, b=0, l=0, r=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    extra    = [target_col] if target_col in df.columns else []
    num_cols = num_df.columns.tolist()
    if len(num_cols) >= 2:
        corr_df = df[num_cols + extra].copy()
        if extra:
            corr_df[extra[0]] = encode_target(df[extra[0]])
        corr = corr_df.corr(numeric_only=True)
        fig_hm = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Correlation heatmap",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_hm, use_container_width=True)