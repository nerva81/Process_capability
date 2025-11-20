import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backend.csv_utils import detect_csv_format, force_numeric  # <-- backend utils

st.set_page_config(page_title="Pareto Analysis", layout="wide")

st.title("Pareto Analysis of Categories")

st.markdown(
    """
1. Upload a dataset (CSV or Excel).  
2. Select the **category column** and one or more **value columns**.  
3. Choose a Pareto ratio (e.g., 80:20).  
4. The chart will display:  
   - bars = absolute values per category  
   - line = cumulative %  
   - dashed line = chosen Pareto threshold  
"""
)

# ---------------- File loading ----------------

def load_file(uploaded_file) -> pd.DataFrame:
    """Loads CSV or Excel with smart CSV detection."""
    name = uploaded_file.name.lower()

    # Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    # CSV / TXT – detect delimiter & decimal
    raw_bytes = uploaded_file.read()
    delimiter, decimal = detect_csv_format(raw_bytes)

    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=delimiter,
        decimal=decimal,
    )
    return df


# ---------------- Pareto computation ----------------

def compute_pareto(df: pd.DataFrame, category_col: str, value_col: str):
    """Computes Pareto distribution for a column."""
    df_num = df.copy()
    # robust numeric conversion (handles , as decimal etc.)
    df_num[value_col] = force_numeric(df_num[value_col]).fillna(0)

    grouped = df_num.groupby(category_col)[value_col].sum().reset_index()
    grouped = grouped.sort_values(value_col, ascending=False)

    total = grouped[value_col].sum()

    if total == 0:
        grouped["share"] = 0
        grouped["cum_share"] = 0
    else:
        grouped["share"] = grouped[value_col] / total
        grouped["cum_share"] = grouped["share"].cumsum()

    return grouped, total


# ---------------- Chart (Plotly) ----------------

def pareto_chart(pareto_df: pd.DataFrame,
                 category_col: str,
                 value_col: str,
                 value_title: str,
                 threshold: float):

    if pareto_df.empty:
        return None

    df = pareto_df.copy()
    df["cum_share_pct"] = df["cum_share"] * 100

    categories = df[category_col].tolist()
    values = df[value_col].tolist()
    cumulative = df["cum_share_pct"].tolist()

    fig = go.Figure()

    # 1) Bars – absolute values
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            name=value_title,
            marker_color="indianred",
            opacity=0.9,
        )
    )

    # 2) Cumulative line – %
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=cumulative,
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color="steelblue", width=3),
            yaxis="y2",
        )
    )

    # 3) Horizontal Pareto threshold line – %
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[threshold * 100] * len(categories),
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name=f"Threshold ({int(threshold*100)} %)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Pareto Chart",
        xaxis=dict(
            title=category_col,
            tickangle=-90,  # vertical labels
        ),
        yaxis=dict(
            title=value_title,
            showgrid=True,
        ),
        yaxis2=dict(
            title="Cumulative (%)",
            overlaying="y",
            side="right",
            range=[0, 110],
            tickformat=".0f",
        ),
        legend=dict(
            orientation="h",
            y=-0.2,
        ),
        bargap=0.1,
        height=600,
    )

    return fig


# ---------------- Main UI ----------------

uploaded_file = st.file_uploader(
    "Upload a data file (CSV, TXT, XLSX)",
    type=["csv", "txt", "xlsx"]
)

if uploaded_file is None:
    st.info("⬆️ Please upload a file to begin.")
    st.stop()

df = load_file(uploaded_file)

if df.empty:
    st.error("The file is empty or could not be read.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)

columns = list(df.columns)

if len(columns) < 2:
    st.error("You need at least one category column and one value column.")
    st.stop()

# ---------------- User selections ----------------

st.markdown("### Analysis settings")

col1, col2 = st.columns(2)

with col1:
    category_col = st.selectbox(
        "Select the **category column**:",
        options=columns
    )

with col2:
    numeric_cols = [
        c for c in columns
        if pd.api.types.is_numeric_dtype(df[c])
        or force_numeric(df[c]).notna().any()   # use the same numeric logic
    ]
    if not numeric_cols:
        numeric_cols = columns

    value_cols = st.multiselect(
        "Select **value columns** (you may choose multiple):",
        options=columns,
        default=numeric_cols[:1],
    )

if not value_cols:
    st.warning("Select at least one value column.")
    st.stop()

# ---------------- Pareto ratio selection ----------------

st.markdown("### Pareto ratio")

pareto_options = {
    "60 : 40": 0.60,
    "65 : 35": 0.65,
    "70 : 30": 0.70,
    "75 : 25": 0.75,
    "80 : 20": 0.80,
    "85 : 15": 0.85,
    "90 : 10": 0.90,
}

ratio_label = st.selectbox("Choose Pareto ratio:", list(pareto_options.keys()), index=4)
threshold = pareto_options[ratio_label]

st.markdown(
    f"Selected ratio: **{ratio_label}** → categories up to **{threshold*100:.0f}%** cumulative share."
)

st.markdown("---")

# ---------------- Generate charts ----------------

tabs = st.tabs(value_cols)

for tab, col_name in zip(tabs, value_cols):
    with tab:
        st.markdown(f"### Value column: `{col_name}`")

        pareto_df, total = compute_pareto(df, category_col, col_name)

        if total == 0:
            st.warning("This column contains only zeros or non-numeric values.")
            continue

        fig = pareto_chart(
            pareto_df,
            category_col,
            col_name,
            value_title=f"Value ({col_name})",
            threshold=threshold,
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Table output
        st.markdown("#### Pareto table")
        table = pareto_df.copy()
        table["Share [%]"] = (table["share"] * 100).round(1)
        table["Cumulative [%]"] = (table["cum_share"] * 100).round(1)
        table["Within ratio"] = table["cum_share"] <= threshold

        st.dataframe(
            table[[category_col, col_name, "Share [%]", "Cumulative [%]", "Within ratio"]],
            use_container_width=True,
        )
