import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

from backend.csv_utils import detect_csv_format, force_numeric
from backend.msa_core import (
    cohen_kappa,
    fleiss_kappa_from_long,
    compute_grr_anova_core,
    compute_grr_xbar_core,
)


# ==========================
# Helper functions – loading & common UI
# ==========================


def load_msa_csv(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None

    raw_bytes = upload.read()
    if not raw_bytes:
        return None

    delimiter, decimal = detect_csv_format(raw_bytes)
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=delimiter,
        decimal=decimal,
        engine="python",
    )
    return df


def select_basic_columns(df: pd.DataFrame, numeric_only: bool = True):
    st.subheader("Column mapping")

    cols = list(df.columns)
    col_part = st.selectbox("Part column", cols, key="msa_part_col")
    col_operator = st.selectbox("Operator / appraiser column", cols, key="msa_op_col")

    if numeric_only:
        numeric_candidates = [
            c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or c not in (col_part, col_operator)
        ]
        if not numeric_candidates:
            numeric_candidates = cols
        col_measure = st.selectbox("Measurement column", numeric_candidates, key="msa_meas_col")
    else:
        col_measure = st.selectbox("Result / category column", cols, key="msa_attr_col")

    return col_part, col_operator, col_measure


def show_basic_preview(df: pd.DataFrame):
    with st.expander("Preview data", expanded=False):
        st.write("Shape:", df.shape)
        st.dataframe(df.head(20))


# ==========================
# Type 1 – bias & repeatability
# ==========================


def render_type1_bias_repeatability(df: pd.DataFrame):
    st.header("Type 1 – Bias & Repeatability")

    cols = list(df.columns)
    col_measure = st.selectbox("Measurement column", cols, key="type1_meas_col")
    measure = force_numeric(df[col_measure].copy()).dropna()

    if measure.empty:
        st.warning("No numeric data in selected column.")
        return

    reference = st.number_input("Reference (master) value", value=float(measure.mean()))
    tolerance = st.number_input("Total tolerance (USL-LSL)", value=float(measure.std() * 10), min_value=0.0)

    st.write("---")

    n = len(measure)
    avg = measure.mean()
    s = measure.std(ddof=1)

    bias = avg - reference
    bias_pct_tol = 100 * bias / tolerance if tolerance > 0 else np.nan
    repeatability = s
    grr = repeatability  # Type 1 – only EV
    pct_tol_grr = 100 * 6 * grr / tolerance if tolerance > 0 else np.nan
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=avg, scale=s / np.sqrt(n)) if n > 1 else (np.nan, np.nan)

    c_pk_equiv = (tolerance / 2) / (3 * s) if s > 0 else np.nan

    c1, c2 = st.columns(2)
    with c1:
        st.metric("n", n)
        st.metric("Mean", f"{avg:.5g}")
        st.metric("Std (repeatability)", f"{s:.5g}")
    with c2:
        st.metric("Bias", f"{bias:.5g}")
        st.metric("%Tolerance of Bias", f"{bias_pct_tol:.1f} %")
        st.metric("Cpk (approx. from Type1)", f"{c_pk_equiv:.2f}" if not np.isnan(c_pk_equiv) else "N/A")

    st.write("#### Bias confidence interval (95%)")
    st.write(f"{ci_low:.5g} … {ci_high:.5g}")

    fig, ax = plt.subplots()
    ax.hist(measure, bins="auto", alpha=0.7)
    ax.axvline(reference, color="tab:red", linestyle="--", label="Reference")
    ax.axvline(avg, color="tab:blue", linestyle="-", label="Mean")
    ax.set_xlabel(col_measure)
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.info(
        "Rule of thumb: |Bias| < 10% of tolerance and Cpk > 1.33 is usually considered acceptable for Type 1."
    )


# ==========================
# Type 2 – variable Gage R&R (ANOVA / X̄–R)
# ==========================


def render_type2_variable(df: pd.DataFrame, method: str):
    if method == "ANOVA":
        st.header("Type 2 – Variable Gage R&R (ANOVA)")
    else:
        st.header("Type 2 – Variable Gage R&R (X̄–R)")

    col_part, col_operator, col_measure = select_basic_columns(df, numeric_only=True)

    data = df[[col_part, col_operator, col_measure]].copy()
    data.columns = ["Part", "Operator", "Measurement"]
    data["Measurement"] = force_numeric(data["Measurement"])
    data = data.dropna()

    if data.empty:
        st.warning("No valid numeric data after cleaning.")
        return

    tolerance = st.number_input("Tolerance (USL - LSL)", value=float(data["Measurement"].std() * 10), min_value=0.0)

    if method == "ANOVA":
        results = compute_grr_anova_core(data, tol=tolerance)
    else:
        try:
            results = compute_grr_xbar_core(data, tol=tolerance)
        except ValueError as e:
            st.error(str(e))
            return

    st.subheader("Variance components")

    comp_df = pd.DataFrame(
        {
            "Component": ["Repeatability (EV)", "Reproducibility", "Gage R&R", "Part-to-part", "Total"],
            "Std dev": [
                results["EV"],
                results["Reprod"],
                results["GRR"],
                results["PV"],
                results["TV"],
            ],
            "%Study variation": [
                results["%SV_EV"],
                results["%SV_Reprod"],
                results["%SV_GRR"],
                results["%SV_PV"],
                results["%SV_TV"],
            ],
            "%Tolerance": [
                results["%Tol_EV"],
                results["%Tol_Reprod"],
                results["%Tol_GRR"],
                results["%Tol_PV"],
                results["%Tol_TV"],
            ],
        }
    )

    st.dataframe(comp_df.style.format({"Std dev": "{:.5g}", "%Study variation": "{:.1f}", "%Tolerance": "{:.1f}"}))

    st.metric("Number of distinct categories (ndc)", f"{results['ndc']:.1f}" if not np.isnan(results["ndc"]) else "N/A")

    st.write("---")
    st.subheader("Boxplots by part and operator")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        data.boxplot(column="Measurement", by="Part", ax=ax)
        ax.set_title("By part")
        ax.set_xlabel("Part")
        ax.set_ylabel("Measurement")
        plt.suptitle("")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        data.boxplot(column="Measurement", by="Operator", ax=ax)
        ax.set_title("By operator")
        ax.set_xlabel("Operator")
        ax.set_ylabel("Measurement")
        plt.suptitle("")
        st.pyplot(fig)

    st.info(
        "Typical acceptance criteria: %GRR (of study variation) ≤ 30% and ndc ≥ 5. "
        "But always check context of your process."
    )


# ==========================
# Type 3 – stability / drift
# ==========================


def render_type3_stability(df: pd.DataFrame):
    st.header("Type 3 – Stability / Drift")

    cols = list(df.columns)
    col_measure = st.selectbox("Measurement column", cols, key="type3_meas_col")
    col_time = st.selectbox(
        "Time / sequence column (optional)",
        ["<index>"] + cols,
        index=0,
        key="type3_time_col",
    )

    y = force_numeric(df[col_measure].copy()).dropna()
    if y.empty:
        st.warning("No numeric data.")
        return

    if col_time == "<index>":
        x = np.arange(len(y))
        x_label = "Index"
    else:
        x = pd.to_datetime(df.loc[y.index, col_time], errors="coerce")
        if x.isna().all():
            x = np.arange(len(y))
            x_label = "Index"
        else:
            x_label = col_time

    LSL = st.number_input("LSL (optional)", value=float(y.mean() - 3 * y.std()), key="type3_lsl")
    USL = st.number_input("USL (optional)", value=float(y.mean() + 3 * y.std()), key="type3_usl")

    # Regression for drift
    x_num = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y.values)

    st.write("#### Trend")
    st.write(f"Slope: {slope:.5g} per sample, p-value: {p_value:.3g}")

    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", linestyle="-", label="Measurement")
    ax.axhline(y.mean(), linestyle="--", label="Mean")
    ax.axhline(LSL, linestyle="--", label="LSL")
    ax.axhline(USL, linestyle="--", label="USL")
    ax.set_xlabel(x_label)
    ax.set_ylabel(col_measure)

    # regression line on numeric index but mapped to x positions
    x_reg = np.linspace(0, len(y) - 1, 100)
    y_reg = intercept + slope * x_reg
    if isinstance(x, pd.Series) and np.issubdtype(x.dtype, np.datetime64):
        # map numeric regression x to datetime range
        x_time = pd.to_datetime(
            np.linspace(x.iloc[0].value, x.iloc[-1].value, 100)
        )
        ax.plot(x_time, y_reg, linestyle=":", label="Trend")
    else:
        ax.plot(x_reg, y_reg, linestyle=":", label="Trend")
    ax.legend()
    st.pyplot(fig)

    st.info(
        "If the trend slope is statistically significant (small p-value) and moves towards spec limits, "
        "measurement system may be unstable."
    )


# ==========================
# Attribute MSA
# ==========================


def render_attribute_msa(df: pd.DataFrame):
    st.header("Attribute MSA")

    cols = list(df.columns)
    col_part = st.selectbox("Part column", cols, key="attr_part_col")
    col_rater = st.selectbox("Appraiser column", cols, key="attr_rater_col")
    col_rating = st.selectbox("Result / category column", cols, key="attr_rating_col")

    has_standard = st.checkbox("Data include reference / standard column?")
    col_ref = None
    if has_standard:
        col_ref = st.selectbox("Reference column", cols, key="attr_ref_col")

    data = df[[col_part, col_rater, col_rating] + ([col_ref] if col_ref else [])].copy()
    data.columns = ["Part", "Rater", "Rating"] + (["Reference"] if col_ref else [])

    # Agreement with reference
    if col_ref:
        df_ref = data.dropna(subset=["Rating", "Reference"]).copy()
        if df_ref.empty:
            st.warning("No valid data with reference.")
        else:
            st.subheader("Agreement with reference")
            df_ref["match"] = df_ref["Rating"].astype(str) == df_ref["Reference"].astype(str)
            overall = df_ref["match"].mean() * 100
            st.metric("Overall % agreement with reference", f"{overall:.1f}%")

            by_rater = df_ref.groupby("Rater")["match"].mean().sort_values(ascending=False) * 100
            st.write("Agreement by appraiser:")
            st.dataframe(by_rater.to_frame("%Agreement"))

    # Between-appraiser agreement
    st.subheader("Between-appraiser agreement (kappa)")

    df_long = data.dropna(subset=["Part", "Rater", "Rating"]).copy()
    if df_long.empty:
        st.warning("No valid attribute data.")
        return

    # pairwise Cohen kappa
    raters = sorted(df_long["Rater"].unique())
    records = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            r1, r2 = raters[i], raters[j]
            sub = df_long[df_long["Rater"].isin([r1, r2])]
            pivot = sub.pivot_table(index="Part", columns="Rater", values="Rating", aggfunc="first").dropna()
            if pivot.shape[0] == 0:
                continue
            k = cohen_kappa(pivot[r1], pivot[r2])
            records.append({"Rater 1": r1, "Rater 2": r2, "Cohen κ": k})

    if records:
        k_df = pd.DataFrame(records)
        st.dataframe(k_df.style.format({"Cohen κ": "{:.3f}"}))
        st.write(f"Average pairwise κ: {np.nanmean(k_df['Cohen κ']):.3f}")
    else:
        st.info("Not enough overlapping parts between raters to compute pairwise Cohen κ.")

    # Fleiss kappa
    fleiss_k = fleiss_kappa_from_long(df_long["Part"], df_long["Rater"], df_long["Rating"])
    if not np.isnan(fleiss_k):
        st.metric("Fleiss κ (all appraisers)", f"{fleiss_k:.3f}")
    else:
        st.info("Fleiss κ could not be computed (insufficient data).")


# ==========================
# Main page
# ==========================


def main():
    st.title("📏 Measurement System Analysis (MSA)")

    msa_type = st.selectbox(
        "Select MSA type",
        [
            "Type 1 – Bias & Repeatability",
            "Type 2 – Variable (ANOVA)",
            "Type 2 – Variable (X̄–R)",
            "Type 3 – Stability / Drift",
            "Attribute MSA",
        ],
    )

    upload = st.file_uploader("Upload CSV file with MSA data", type=["csv", "txt"])
    if upload is None:
        st.info("Please upload a CSV file to continue.")
        return

    df = load_msa_csv(upload)
    if df is None or df.empty:
        st.error("Could not load data or file is empty.")
        return

    show_basic_preview(df)

    if msa_type.startswith("Type 1"):
        render_type1_bias_repeatability(df)
    elif msa_type == "Type 2 – Variable (ANOVA)":
        render_type2_variable(df, method="ANOVA")
    elif msa_type == "Type 2 – Variable (X̄–R)":
        render_type2_variable(df, method="X̄–R")
    elif msa_type.startswith("Type 3"):
        render_type3_stability(df)
    else:
        render_attribute_msa(df)


if __name__ == "__main__":
    main()
