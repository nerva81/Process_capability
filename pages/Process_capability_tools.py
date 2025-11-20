import math
import io
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

from backend.csv_utils import detect_csv_format, force_numeric
from backend.capability_core import compute_ewma, compute_cusum


# ==========================
# Helper functions – loading & common UI
# ==========================


def load_capability_csv(upload) -> Optional[pd.DataFrame]:
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


def show_basic_preview(df: pd.DataFrame):
    with st.expander("Preview data", expanded=False):
        st.write("Shape:", df.shape)
        st.dataframe(df.head(20))


def select_value_and_group_columns(df: pd.DataFrame):
    cols = list(df.columns)
    col_value = st.selectbox("Value / measurement column", cols, key="cap_value_col")

    group_col_options = ["<none>"] + cols
    col_group = st.selectbox(
        "Subgroup column (optional – for X̄–R / within variation)",
        group_col_options,
        index=0,
        key="cap_group_col",
    )

    time_col = st.selectbox(
        "Time / sequence column (optional – for time charts)",
        ["<index>"] + cols,
        index=0,
        key="cap_time_col",
    )

    return col_value, (None if col_group == "<none>" else col_group), time_col


# ==========================
# Capability calculations
# ==========================


def compute_within_std(values: pd.Series, group: Optional[pd.Series]) -> float:
    """
    Simple within-subgroup standard deviation. If no subgroup, falls back to overall std.
    """
    v = values.dropna().astype(float)
    if v.empty:
        return np.nan

    if group is None:
        return v.std(ddof=1)

    df = pd.DataFrame({"v": v, "g": group})
    groups = df.dropna(subset=["g"]).groupby("g")["v"]

    # Rbar / d2 for average subgroup size
    sizes = groups.size().values
    if len(sizes) == 0:
        return v.std(ddof=1)

    ranges = groups.max() - groups.min()
    Rbar = ranges.mean()
    avg_n = sizes.mean()

    # approximate d2 as function of n for n up to ~25, fallback
    d2_table = {
        2: 1.128,
        3: 1.693,
        4: 2.059,
        5: 2.326,
        6: 2.534,
        7: 2.704,
        8: 2.847,
        9: 2.97,
        10: 3.078,
    }
    n_round = int(round(avg_n))
    d2 = d2_table.get(n_round, 3.0)
    if d2 <= 0 or np.isnan(Rbar):
        return v.std(ddof=1)
    return Rbar / d2


def compute_capability_indices(
    values: pd.Series,
    LSL: Optional[float],
    USL: Optional[float],
    stdev_within: float,
) -> dict:
    x = values.dropna().astype(float)
    if x.empty:
        return {}

    mean = x.mean()
    stdev_overall = x.std(ddof=1)

    cp = cpk = pp = ppk = np.nan

    if LSL is not None and USL is not None and USL > LSL:
        cp = (USL - LSL) / (6 * stdev_within) if stdev_within > 0 else np.nan
        pp = (USL - LSL) / (6 * stdev_overall) if stdev_overall > 0 else np.nan
        cpu = (USL - mean) / (3 * stdev_within) if stdev_within > 0 else np.nan
        cpl = (mean - LSL) / (3 * stdev_within) if stdev_within > 0 else np.nan
        cpk = min(cpu, cpl)
        ppu = (USL - mean) / (3 * stdev_overall) if stdev_overall > 0 else np.nan
        ppl = (mean - LSL) / (3 * stdev_overall) if stdev_overall > 0 else np.nan
        ppk = min(ppu, ppl)

    return {
        "mean": mean,
        "stdev_overall": stdev_overall,
        "stdev_within": stdev_within,
        "Cp": cp,
        "Cpk": cpk,
        "Pp": pp,
        "Ppk": ppk,
    }


# ==========================
# Plot helpers
# ==========================


def plot_histogram_with_specs(x, LSL, USL, title, xlabel):
    fig, ax = plt.subplots()
    ax.hist(x, bins="auto", alpha=0.7)
    if LSL is not None:
        ax.axvline(LSL, linestyle="--", label="LSL")
    if USL is not None:
        ax.axvline(USL, linestyle="--", label="USL")
    ax.axvline(np.mean(x), linestyle="-", label="Mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def plot_time_series(x, y, LSL, USL, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", linestyle="-")
    if LSL is not None:
        ax.axhline(LSL, linestyle="--", label="LSL")
    if USL is not None:
        ax.axhline(USL, linestyle="--", label="USL")
    ax.axhline(np.mean(y), linestyle="-.", label="Mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)


def plot_xbar_r_chart(df: pd.DataFrame, value_col: str, group_col: str):
    st.subheader("X̄–R chart by subgroup")

    grouped = df.groupby(group_col)[value_col]
    sizes = grouped.size()

    if sizes.nunique() != 1:
        st.warning("For a classic X̄–R chart, all subgroups should have the same size.")
    n = int(round(sizes.mean()))

    d2_table = {
        2: 1.128,
        3: 1.693,
        4: 2.059,
        5: 2.326,
        6: 2.534,
        7: 2.704,
        8: 2.847,
        9: 2.97,
        10: 3.078,
    }
    d2 = d2_table.get(n, 3.0)

    means = grouped.mean()
    ranges = grouped.max() - grouped.min()
    Rbar = ranges.mean()
    Xbarbar = means.mean()

    sigma_within = Rbar / d2 if d2 > 0 else np.nan
    UCL_x = Xbarbar + 3 * sigma_within / (n ** 0.5)
    LCL_x = Xbarbar - 3 * sigma_within / (n ** 0.5)

    UCL_r = Rbar * 2.574  # approx for n≈5
    LCL_r = 0.0

    # Xbar chart
    fig1, ax1 = plt.subplots()
    ax1.plot(means.index.astype(str), means.values, marker="o")
    ax1.axhline(Xbarbar, linestyle="-", label="CL")
    ax1.axhline(UCL_x, linestyle="--", label="UCL")
    ax1.axhline(LCL_x, linestyle="--", label="LCL")
    ax1.set_xlabel(group_col)
    ax1.set_ylabel("Group mean")
    ax1.legend()
    st.pyplot(fig1)

    # R chart
    fig2, ax2 = plt.subplots()
    ax2.plot(ranges.index.astype(str), ranges.values, marker="o")
    ax2.axhline(Rbar, linestyle="-", label="CL")
    ax2.axhline(UCL_r, linestyle="--", label="UCL")
    ax2.axhline(LCL_r, linestyle="--", label="LCL")
    ax2.set_xlabel(group_col)
    ax2.set_ylabel("Range")
    ax2.legend()
    st.pyplot(fig2)

    return sigma_within


# ==========================
# Main page
# ==========================


def main():
    st.title("📊 Process Capability Tools")

    upload = st.file_uploader("Upload CSV file with process data", type=["csv", "txt"])
    if upload is None:
        st.info("Please upload a CSV file to continue.")
        return

    df = load_capability_csv(upload)
    if df is None or df.empty:
        st.error("Could not load data or file is empty.")
        return

    show_basic_preview(df)

    # NEW: možnost mít hodnoty ve více sloupcích (subgroup size = počet sloupců)
    multi_mode = st.checkbox(
        "Values are in multiple columns (each row = one subgroup, each column = one measurement in subgroup)",
        value=False,
    )

    # ======================
    # SINGLE-COLUMN MODE
    # ======================
    if not multi_mode:
        col_value, col_group, col_time = select_value_and_group_columns(df)

        x = force_numeric(df[col_value].copy())
        x = x.dropna()
        if x.empty:
            st.warning("No numeric data in selected value column.")
            return

        # Specification limits
        st.subheader("Specification")
        c1, c2, c3 = st.columns(3)
        with c1:
            has_lsl = st.checkbox("Use LSL")
        with c2:
            has_usl = st.checkbox("Use USL", value=True)
        with c3:
            target = st.number_input("Target (optional)", value=float(x.mean()))

        LSL = st.number_input("LSL", value=float(x.mean() - 3 * x.std()), key="cap_lsl") if has_lsl else None
        USL = st.number_input("USL", value=float(x.mean() + 3 * x.std()), key="cap_usl") if has_usl else None

        # Basic statistics
        st.subheader("Basic statistics")
        st.write(f"n = {len(x)}")
        st.write(f"Mean = {x.mean():.5g}")
        st.write(f"Std (overall) = {x.std(ddof=1):.5g}")

        # Normality test
        if len(x) <= 5000:
            stat, pvalue = shapiro(x)
            st.write(f"Shapiro–Wilk normality p-value: {pvalue:.3g} (n={len(x)})")
        else:
            st.info("Sample too large for Shapiro–Wilk, skipping normality test.")

        # Within std & Xbar-R chart
        st.write("---")
        st.subheader("Within-subgroup variation & control charts")

        sigma_within = compute_within_std(
            df[col_value],
            df[col_group] if col_group is not None else None,
        )
        st.write(
            f"Within-subgroup std ≈ {sigma_within:.5g}" if not np.isnan(sigma_within) else "Within std not available."
        )

        if col_group is not None:
            sigma_within = plot_xbar_r_chart(df.assign(**{col_value: x}), col_value, col_group) or sigma_within

        # Capability indices
        st.write("---")
        st.subheader("Capability indices")

        caps = compute_capability_indices(x, LSL, USL, sigma_within)
        if not caps:
            st.warning("Capability could not be computed (missing data or specs).")
            return

        cap_df = pd.DataFrame(
            {
                "Index": ["Cp", "Cpk", "Pp", "Ppk"],
                "Value": [caps["Cp"], caps["Cpk"], caps["Pp"], caps["Ppk"]],
            }
        )
        st.dataframe(cap_df.style.format({"Value": "{:.3f}"}))

        st.metric("Cpk", f"{caps['Cpk']:.3f}" if not np.isnan(caps["Cpk"]) else "N/A")

        # Histogram
        st.write("---")
        st.subheader("Histogram with specification limits")
        plot_histogram_with_specs(x, LSL, USL, title="Histogram", xlabel=col_value)

        # Time-based charts
        st.write("---")
        st.subheader("Time-based charts (optional)")

        if col_time == "<index>":
            t = np.arange(len(x))
            t_label = "Index"
        else:
            t_full = df.loc[x.index, col_time]
            t = pd.to_datetime(t_full, errors="coerce")
            if t.isna().all():
                t = np.arange(len(x))
                t_label = "Index"
            else:
                t_label = col_time

        plot_time_series(t, x, LSL, USL, xlabel=t_label, ylabel=col_value)

        # EWMA & CUSUM – na jednotlivých hodnotách
        if not np.isnan(sigma_within):
            st.write("#### EWMA chart")
            lam = st.slider("EWMA λ (smoothing factor)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            z, ewma_UCL, ewma_LCL = compute_ewma(x.values, sigma_within, lam=lam, L=3)
            if z is not None:
                fig, ax = plt.subplots()
                ax.plot(t, x, marker="o", linestyle=":", label="Data")
                ax.plot(t, z, marker="o", linestyle="-", label="EWMA")
                ax.axhline(np.mean(x), linestyle="-.", label="Mean")
                ax.axhline(ewma_UCL, linestyle="--", label="UCL")
                ax.axhline(ewma_LCL, linestyle="--", label="LCL")
                ax.set_xlabel(t_label)
                ax.set_ylabel(col_value)
                ax.legend()
                st.pyplot(fig)

            st.write("#### CUSUM chart")
            c_plus, c_minus, h_up, h_down = compute_cusum(x.values, sigma_within, k_mult=0.5, h_mult=5.0)
            if c_plus is not None:
                fig, ax = plt.subplots()
                ax.plot(t, c_plus, marker="o", linestyle="-", label="C+")
                ax.plot(t, c_minus, marker="o", linestyle="-", label="C-")
                ax.axhline(h_up, linestyle="--", label="+h")
                ax.axhline(h_down, linestyle="--", label="-h")
                ax.set_xlabel(t_label)
                ax.set_ylabel("CUSUM")
                ax.legend()
                st.pyplot(fig)

        # Simple decision
        st.write("---")
        st.subheader("Decision helper")
        cust_cpk = st.number_input("Customer-required Cpk", value=1.33)
        if not np.isnan(caps["Cpk"]) and caps["Cpk"] >= cust_cpk:
            st.success(f"✅ Process appears capable (Cpk = {caps['Cpk']:.2f} ≥ {cust_cpk})")
        else:
            st.warning(
                f"⚠️ Process may not be capable (Cpk = {caps['Cpk']:.2f} < {cust_cpk})"
                if not np.isnan(caps["Cpk"])
                else "⚠️ Cpk not available – cannot evaluate capability."
            )

        return  # konec single-column větve

    # ======================
    # MULTI-COLUMN MODE
    # ======================
    cols = list(df.columns)
    value_cols = st.multiselect(
        "Value columns (each column = one measurement in a subgroup)",
        cols,
        default=cols[:2] if len(cols) >= 2 else cols,
        key="cap_value_cols_multi",
    )

    if not value_cols:
        st.warning("Select at least one value column.")
        return

    # Time / sequence column pro subskupiny
    col_time = st.selectbox(
        "Time / sequence column for subgroups (optional)",
        ["<index>"] + cols,
        index=0,
        key="cap_time_col_multi",
    )

    # Převod všech vybraných sloupců na čísla
    values_wide = df[value_cols].apply(force_numeric)
    # long-form pro within std a Xbar-R
    records = []
    for idx, row in values_wide.iterrows():
        subgroup = idx  # jedna řádka = jedna subgroup
        for val in row.values:
            if not pd.isna(val):
                records.append({"subgroup": subgroup, "value": val})

    df_long = pd.DataFrame(records)
    if df_long.empty:
        st.warning("No numeric data in selected value columns.")
        return

    x = df_long["value"].copy()
    x = x.dropna()

    # Specification limits
    st.subheader("Specification")
    c1, c2, c3 = st.columns(3)
    with c1:
        has_lsl = st.checkbox("Use LSL", key="multi_has_lsl")
    with c2:
        has_usl = st.checkbox("Use USL", value=True, key="multi_has_usl")
    with c3:
        target = st.number_input("Target (optional)", value=float(x.mean()), key="multi_target")

    LSL = (
        st.number_input(
            "LSL", value=float(x.mean() - 3 * x.std()), key="cap_lsl_multi"
        )
        if has_lsl
        else None
    )
    USL = (
        st.number_input(
            "USL", value=float(x.mean() + 3 * x.std()), key="cap_usl_multi"
        )
        if has_usl
        else None
    )

    # Basic statistics
    st.subheader("Basic statistics")
    st.write(f"n = {len(x)}")
    st.write(f"Mean = {x.mean():.5g}")
    st.write(f"Std (overall) = {x.std(ddof=1):.5g}")

    # Normality test
    if len(x) <= 5000:
        stat, pvalue = shapiro(x)
        st.write(f"Shapiro–Wilk normality p-value: {pvalue:.3g} (n={len(x)})")
    else:
        st.info("Sample too large for Shapiro–Wilk, skipping normality test.")

    # Within std & Xbar-R chart – jedna subgroup = řádek, subgroup size = počet sloupců
    st.write("---")
    st.subheader("Within-subgroup variation & control charts")

    sigma_within = compute_within_std(df_long["value"], df_long["subgroup"])
    st.write(
        f"Within-subgroup std ≈ {sigma_within:.5g}" if not np.isnan(sigma_within) else "Within std not available."
    )

    sigma_within = plot_xbar_r_chart(df_long, "value", "subgroup") or sigma_within

    # Capability indices
    st.write("---")
    st.subheader("Capability indices")

    caps = compute_capability_indices(x, LSL, USL, sigma_within)
    if not caps:
        st.warning("Capability could not be computed (missing data or specs).")
        return

    cap_df = pd.DataFrame(
        {
            "Index": ["Cp", "Cpk", "Pp", "Ppk"],
            "Value": [caps["Cp"], caps["Cpk"], caps["Pp"], caps["Ppk"]],
        }
    )
    st.dataframe(cap_df.style.format({"Value": "{:.3f}"}))

    st.metric("Cpk", f"{caps['Cpk']:.3f}" if not np.isnan(caps["Cpk"]) else "N/A")

    # Histogram
    st.write("---")
    st.subheader("Histogram with specification limits")
    plot_histogram_with_specs(x, LSL, USL, title="Histogram", xlabel="Value")

    # Time-based charts – na průměrech subskupin
    st.write("---")
    st.subheader("Time-based charts (subgroup means)")

    subgroup_means = values_wide.mean(axis=1, skipna=True)

    if col_time == "<index>":
        t = np.arange(len(subgroup_means))
        t_label = "Index"
    else:
        t_series = pd.to_datetime(df[col_time], errors="coerce")
        if t_series.isna().all():
            t = np.arange(len(subgroup_means))
            t_label = "Index"
        else:
            t = t_series
            t_label = col_time

    plot_time_series(t, subgroup_means, LSL, USL, xlabel=t_label, ylabel="Subgroup mean")

    if not np.isnan(sigma_within):
        st.write("#### EWMA chart (on subgroup means)")
        lam = st.slider(
            "EWMA λ (smoothing factor)", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="ewma_lambda_multi"
        )
        z, ewma_UCL, ewma_LCL = compute_ewma(subgroup_means.values, sigma_within, lam=lam, L=3)
        if z is not None:
            fig, ax = plt.subplots()
            ax.plot(t, subgroup_means, marker="o", linestyle=":", label="Subgroup means")
            ax.plot(t, z, marker="o", linestyle="-", label="EWMA")
            ax.axhline(subgroup_means.mean(), linestyle="-.", label="Mean")
            ax.axhline(ewma_UCL, linestyle="--", label="UCL")
            ax.axhline(ewma_LCL, linestyle="--", label="LCL")
            ax.set_xlabel(t_label)
            ax.set_ylabel("Subgroup mean")
            ax.legend()
            st.pyplot(fig)

        st.write("#### CUSUM chart (on subgroup means)")
        c_plus, c_minus, h_up, h_down = compute_cusum(subgroup_means.values, sigma_within, k_mult=0.5, h_mult=5.0)
        if c_plus is not None:
            fig, ax = plt.subplots()
            ax.plot(t, c_plus, marker="o", linestyle="-", label="C+")
            ax.plot(t, c_minus, marker="o", linestyle="-", label="C-")
            ax.axhline(h_up, linestyle="--", label="+h")
            ax.axhline(h_down, linestyle="--", label="-h")
            ax.set_xlabel(t_label)
            ax.set_ylabel("CUSUM")
            ax.legend()
            st.pyplot(fig)

    # Simple decision
    st.write("---")
    st.subheader("Decision helper")
    cust_cpk = st.number_input("Customer-required Cpk", value=1.33, key="cust_cpk_multi")
    if not np.isnan(caps["Cpk"]) and caps["Cpk"] >= cust_cpk:
        st.success(f"✅ Process appears capable (Cpk = {caps['Cpk']:.2f} ≥ {cust_cpk})")
    else:
        st.warning(
            f"⚠️ Process may not be capable (Cpk = {caps['Cpk']:.2f} < {cust_cpk})"
            if not np.isnan(caps["Cpk"])
            else "⚠️ Cpk not available – cannot evaluate capability."
        )


if __name__ == "__main__":
    main()
