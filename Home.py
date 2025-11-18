import math
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro


# ==========================
# Pomocné funkce
# ==========================

def detect_csv_format(raw_bytes: bytes):
    """Hrubá detekce oddělovače a desetinného znaku z prvních pár řádků."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    sample = "\n".join(lines[:10])

    # Detekce oddělovače
    candidates = [";", ",", "\t"]
    counts = {c: sample.count(c) for c in candidates}
    # Preferuj ; pokud má nějaký výskyt (typický evropský export)
    if counts[";"] > 0:
        delimiter = ";"
    elif counts["\t"] > 0:
        delimiter = "\t"
    elif counts[","] > 0:
        delimiter = ","
    else:
        delimiter = ","  # fallback

    # Detekce desetinného znaku:
    # Pokud oddělovač není čárka a ve vzorku je něco jako 1,2 → používáme čárku
    decimal = "."
    import re
    if delimiter != ",":
        if re.search(r"\d,\d", sample):
            decimal = ","
        else:
            decimal = "."

    # Pokud delimiter == ";" často to znamená evropský formát → default decimal ","
    if delimiter == ";" and decimal == ".":
        decimal = ","

    return delimiter, decimal


def compute_ewma(values, sigma_within, lam=0.2, L=3):
    """EWMA hodnoty a konstantní kontrolní meze."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None

    mean = np.mean(values)
    z = np.zeros_like(values, dtype=float)
    z[0] = mean
    for i in range(1, len(values)):
        z[i] = lam * values[i] + (1 - lam) * z[i - 1]

    # dlouhodobý rozptyl EWMA (pro konstantní meze)
    sigma_z = sigma_within * math.sqrt(lam / (2 - lam))
    UCL = mean + L * sigma_z
    LCL = mean - L * sigma_z
    return z, UCL, LCL


def compute_cusum(values, sigma_within, k_mult=0.5, h_mult=5.0):
    """Jednoduchý dvoustranný CUSUM (C+ a C-)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None, None

    target = np.mean(values)
    k = k_mult * sigma_within
    h = h_mult * sigma_within

    c_plus = np.zeros_like(values, dtype=float)
    c_minus = np.zeros_like(values, dtype=float)

    for i in range(1, len(values)):
        c_plus[i] = max(0, c_plus[i - 1] + values[i] - target - k)
        c_minus[i] = min(0, c_minus[i - 1] + values[i] - target + k)

    return c_plus, c_minus, h, -h


# ==========================
# UI – nadpis a vstupy
# ==========================

st.title("Process capability from CSV")

col1a, col1b, col2a = st.columns(3)
st.header("Input specification limits and customer Cpk requirement")
with col1a:
    LSL = st.number_input("Lower specification limit (LSL)", value=float(1185.20))

with col1b:
    USL = st.number_input("Upper specification limit (USL)", value=float(1188.00))

with col2a:
    cust_cpk = st.selectbox(
        "Customer requirement to Cpk",
        (float(1.67), float(1.33), float(1.00))
    )

uploaded_file = st.file_uploader(
    "Load CSV file (Values in columns or one column under each other)",
    type="csv"
)

if uploaded_file:
    use_header = st.checkbox('CSV file includes header line.', value=True)

    # Načti raw obsah pro autodetekci
    raw_bytes = uploaded_file.getvalue()
    auto_delim, auto_dec = detect_csv_format(raw_bytes)

    st.subheader("CSV format options")
    col_sep, col_dec = st.columns(2)
    with col_sep:
        delimiter = st.selectbox(
            "CSV delimiter",
            options=[";", ",", "\t"],
            index=[";", ",", "\t"].index(auto_delim),
            format_func=lambda x: {";": "Semicolon (;)", ",": "Comma (,)", "\t": "Tab"}[x],
        )
    with col_dec:
        decimal_sep = st.selectbox(
            "Decimal separator",
            options=[",", "."],
            index=[",", "."].index(auto_dec),
        )

    # Znovu vytvořit buffer z raw_bytes (uploaded_file už jsme „vypili“)
    text_io = io.StringIO(raw_bytes.decode("utf-8", errors="ignore"))

    read_kwargs = {
        "sep": delimiter,
        "decimal": decimal_sep,
    }

    if use_header:
        df = pd.read_csv(text_io, **read_kwargs)
    else:
        df = pd.read_csv(text_io, header=None, **read_kwargs)

    st.write("### Loaded data")
    st.dataframe(df)

    # ==========================
    # Výběr číselných sloupců (měření)
    # ==========================
    numeric_df_full = df.select_dtypes(include=["number"])

    if numeric_df_full.shape[1] == 0:
        st.error("CSV neobsahuje žádný číselný sloupec.")
        st.stop()

    st.info("Select numeric columns that contain measurement values (exclude index columns).")

    numeric_cols = list(numeric_df_full.columns)

    # default: pokud víc číselných sloupců, bereme všechny (uživatel může odškrtnout index)
    default_selection = numeric_cols

    selected_cols = st.multiselect(
        "Measurement columns",
        options=numeric_cols,
        default=default_selection
    )

    if len(selected_cols) == 0:
        st.error("Musíš vybrat alespoň jeden číselný sloupec s měřením.")
        st.stop()

    numeric_df = numeric_df_full[selected_cols]

    # ==========================
    # Detekce sloupců s časem / datem
    # ==========================
    st.subheader("Time / date column (optional)")

    time_candidates = []
    parsed_time_columns = {}

    # vezmeme všechny sloupce kromě vybraných měření
    for col in df.columns:
        if col in selected_cols:
            continue
        s = df[col]

        # přeskoč čistě číselné
        if pd.api.types.is_numeric_dtype(s):
            continue

        # zkus konverzi na datetime
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        if parsed.notna().mean() > 0.7:  # aspoň 70 % hodnot vypadá jako datum/čas
            time_candidates.append(col)
            parsed_time_columns[col] = parsed

    time_series = None
    time_values_individuals = None   # čas pro jednotlivá měření (n=1)
    time_subgroups = None            # čas pro podskupiny (n>1)

    if len(time_candidates) > 0:
        time_col_name = st.selectbox(
            "Time column for x-axis (optional)",
            options=["(none)"] + time_candidates,
            index=0
        )
        if time_col_name != "(none)":
            time_series = parsed_time_columns[time_col_name]
    else:
        st.caption("No obvious time/date column detected. X-axis will use index.")

    # ==========================
    # Příprava proměnných
    # ==========================
    subgroup_means = None
    subgroup_ranges = None
    mr = None
    mr_bar = None
    k = None  # number of subgroups
    n = None  # subgroup size

    # ==========================
    # Detekce typu dat (1 sloupec vs. víc sloupců)
    # ==========================
    if numeric_df.shape[1] == 1:
        # --------------------
        # Jeden sloupec hodnot (podskupiny pod sebou)
        # --------------------
        st.info(
            "Detected a single numeric measurement column. "
            "Data are treated as measurements under each other. "
            "Set subgroup size (1 = individual measurements)."
        )

        # uchovej indexy pro zarovnání s časem
        series_col = numeric_df.iloc[:, 0]
        values = series_col.dropna().values
        valid_idx = series_col.dropna().index

        if time_series is not None:
            time_values_individuals = time_series.loc[valid_idx].values

        if len(values) < 2:
            st.error("Pro analýzu je potřeba alespoň 2 hodnoty.")
            st.stop()

        subgroup_size = st.number_input(
            "Subgroup size (1 = individuals, >1 = measurements per subgroup)",
            min_value=1,
            step=1,
            value=1
        )

        if subgroup_size == 1:
            # -------------------------------------------
            # Subgroup size = 1 → různé metody sigma within
            # + I-MR podklad pro kontrolní diagram z dat

            # -------------------------------------------
            all_values = values
            k = len(all_values)
            n = 1

            if len(all_values) < 2:
                st.error("Pro subgroup size = 1 je potřeba alespoň 2 hodnoty (kvůli moving range).")
                st.stop()

            sigma_method = st.selectbox(
                "Method for within sigma (subgroup size = 1)",
                (
                    "I-MR (moving range)",
                    "Overall standard deviation",
                    "Robust (MAD-based)"
                )
            )

            if sigma_method == "I-MR (moving range)":
                mr = np.abs(np.diff(all_values))
                mr_bar = np.mean(mr)
                d2 = 1.128  # konstanta pro MR ze 2 hodnot
                stdev_within = mr_bar / d2

            elif sigma_method == "Overall standard deviation":
                stdev_within = np.std(all_values, ddof=1)
                mr = np.abs(np.diff(all_values))
                mr_bar = np.mean(mr)

            elif sigma_method == "Robust (MAD-based)":
                median_val = np.median(all_values)
                mad = np.median(np.abs(all_values - median_val))
                stdev_within = mad / 0.6745
                mr = np.abs(np.diff(all_values))
                mr_bar = np.mean(mr)

        else:
            # -----------------------------------
            # Subgroup size > 1 – podskupiny pod sebou
            # -----------------------------------
            num_groups = len(values) // subgroup_size

            if num_groups < 2:
                st.error(
                    "Pro zadanou subgroup size není dostatek dat na alespoň 2 podskupiny. "
                    f"Počet hodnot: {len(values)}, subgroup size: {subgroup_size}."
                )
                st.stop()

            trimmed = values[:num_groups * subgroup_size]
            data_matrix = trimmed.reshape(num_groups, subgroup_size)

            all_values = trimmed
            k = num_groups
            n = subgroup_size

            subgroup_means = data_matrix.mean(axis=1)
            subgroup_ranges = data_matrix.max(axis=1) - data_matrix.min(axis=1)

            within_ss = ((data_matrix - subgroup_means[:, None]) ** 2).sum()
            stdev_within = np.sqrt(within_ss / (k * (n - 1)))

            # čas pro podskupiny – vezmeme první čas v každé podskupině
            if time_series is not None and time_values_individuals is not None:
                # time_values_individuals má stejnou délku jako values
                trimmed_time = time_values_individuals[:num_groups * subgroup_size]
                time_subgroups = trimmed_time[0::subgroup_size]

    else:
        # --------------------------------------
        # Více sloupců – podskupiny ve sloupcích vedle sebe
        # --------------------------------------
        numeric_values = numeric_df.to_numpy()
        k = numeric_df.shape[0]   # počet řádků = počet podskupin
        n = numeric_df.shape[1]   # počet sloupců = velikost podskupiny

        all_values = numeric_values.flatten()

        subgroup_means = numeric_values.mean(axis=1)
        subgroup_ranges = numeric_values.max(axis=1) - numeric_values.min(axis=1)

        within_ss = ((numeric_values - subgroup_means[:, None]) ** 2).sum()
        stdev_within = np.sqrt(within_ss / (k * (n - 1)))

        # čas pro podskupiny – podle řádků
        if time_series is not None:
            time_subgroups = time_series.values

    # ==========================
    # Společné výpočty
    # ==========================
    stdev_overall = np.std(all_values, ddof=1)
    overall_mean = np.mean(all_values)
    overall_median = np.median(all_values)

    # Normalita
    col5, col6 = st.columns(2)
    with col5:
        st.write("### Q–Q plot (Normality test)")
        fig3, ax3 = plt.subplots()
        sm.qqplot(all_values, line='s', ax=ax3)
        st.pyplot(fig3)

    with col6:
        st.write("### Normality test (Shapiro-Wilk)")
        sample = all_values
        if len(all_values) > 5000:
            sample = np.random.choice(all_values, size=5000, replace=False)

        shapiro_stat, shapiro_p = shapiro(sample)
        st.write(f"**W-statistik**: {shapiro_stat:.4f}")
        st.write(f"**p-Value (H0)**: {shapiro_p:.4f}")

        if shapiro_p > 0.05:
            st.success("Data could be evaluated as normally distributed (p > 0.05).")
        else:
            st.warning("Data could NOT be evaluated as normally distributed (p ≤ 0.05). Data must be transformed for capability evaluation.")

    # Cp / Cpk (within) a Pp / Ppk (overall)
    Cp = (USL - LSL) / (6 * stdev_within)
    Cpu = (USL - overall_mean) / (3 * stdev_within)
    Cpl = (overall_mean - LSL) / (3 * stdev_within)
    Cpk = min(Cpu, Cpl)

    Pp = (USL - LSL) / (6 * stdev_overall)
    Ppu = (USL - overall_mean) / (3 * stdev_overall)
    Ppl = (overall_mean - LSL) / (3 * stdev_overall)
    Ppk = min(Ppu, Ppl)

    # ==========================
    # Výsledky + histogram + time plot
    # ==========================
    st.header("Results:")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Median", f"{overall_median:.2f}")
        st.header('Long-term process capability (Pp, Ppk):')
        st.metric("StDev (overall)", f"{stdev_overall:.6f}")
        st.metric("Pp", f"{Pp:.4f}")
        st.metric("Ppl", f"{Ppl:.4f}")
        st.metric("Ppu", f"{Ppu:.4f}")
        st.metric("Ppk", f"{Ppk:.4f}")

        st.write("### Histogram")
        fig, ax = plt.subplots()
        bins_quantity = int(math.sqrt(len(all_values)))
        ax.hist(all_values, bins=bins_quantity, edgecolor='black')
        ax.axvline(USL, linestyle='--', label='USL')
        ax.axvline(LSL, linestyle='--', label='LSL')
        ax.axvline(overall_mean, linestyle='-', label='Mean')
        ax.legend()
        st.pyplot(fig)

    with col4:
        st.metric("Average", f"{overall_mean:.6f}")
        st.header('Short-term process capability (Cp, Cpk):')
        st.metric("StDev (within)", f"{stdev_within:.6f}")
        st.metric("Cpl", f"{Cpl:.4f}")
        st.metric("Cpu", f"{Cpu:.4f}")
        st.metric("Cp", f"{Cp:.4f}")
        st.metric("Cpk", f"{Cpk:.4f}")

        st.write("### Values time plot")
        fig2, ax2 = plt.subplots()

        # pokud máme čas pro jednotlivce, použijeme ho; jinak index
        if n == 1 and time_values_individuals is not None and len(time_values_individuals) == len(all_values):
            x_axis = time_values_individuals
            ax2.set_xlabel("Time")
            ax2.plot(x_axis, all_values, marker='o', linestyle='-')
        elif n > 1 and time_subgroups is not None and subgroup_means is not None and len(time_subgroups) == len(subgroup_means):
            x_axis = time_subgroups
            ax2.set_xlabel("Time (subgroup)")
            ax2.plot(x_axis, subgroup_means, marker='o', linestyle='-')
        else:
            x_axis = range(len(all_values))
            ax2.set_xlabel("Measurement index")
            ax2.plot(x_axis, all_values, marker='o', linestyle='-')

        ax2.axhline(overall_mean, linestyle='--', label='Mean')
        ax2.axhline(USL, linestyle='--', label='USL')
        ax2.axhline(LSL, linestyle='--', label='LSL')
        ax2.set_ylabel("Value")
        ax2.legend()
        st.pyplot(fig2)

    # ==========================
    # Kontrolní diagramy + meze (X / X-bar, R/MR)
    # ==========================
    st.header("Control charts")

    x_values = None
    x_center = None
    x_UCL = None
    x_LCL = None

    r_values = None
    r_center = None
    r_UCL = None
    r_LCL = None

    # Konstanty pro X-bar & R
    control_constants = {
        2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
        3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
        4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
        5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
        6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
        7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
        8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
        9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
        10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
    }

    if n == 1:
        # Individuals (X) chart + MR
        individuals = all_values

        if len(individuals) >= 2:
            if mr is None or mr_bar is None:
                mr = np.abs(np.diff(individuals))
                mr_bar = np.mean(mr)

        x_values = individuals
        x_center = np.mean(individuals)
        x_UCL = x_center + 3 * stdev_within
        x_LCL = x_center - 3 * stdev_within

        # X chart – osa X podle času, pokud máme
        fig_x, ax_x = plt.subplots()
        if time_values_individuals is not None and len(time_values_individuals) == len(individuals):
            x_axis = time_values_individuals
            ax_x.set_xlabel("Time")
        else:
            x_axis = range(len(individuals))
            ax_x.set_xlabel("Observation")

        ax_x.plot(x_axis, individuals, marker='o', linestyle='-')
        ax_x.axhline(x_center, linestyle='-', label='CL (mean)')
        ax_x.axhline(x_UCL, linestyle='--', label='UCL')
        ax_x.axhline(x_LCL, linestyle='--', label='LCL')
        ax_x.set_title("Individuals (X) chart")
        ax_x.set_ylabel("Value")
        ax_x.legend()
        st.pyplot(fig_x)

        # MR chart – čas tady dává menší smysl, necháme index
        if mr is not None and len(mr) > 0:
            D4_MR = 3.267
            UCLmr = D4_MR * mr_bar
            LCLmr = 0.0

            r_values = mr
            r_center = mr_bar
            r_UCL = UCLmr
            r_LCL = LCLmr

            fig_mr, ax_mr = plt.subplots()
            ax_mr.plot(range(len(mr)), mr, marker='o', linestyle='-')
            ax_mr.axhline(mr_bar, linestyle='-', label='CL (MR mean)')
            ax_mr.axhline(UCLmr, linestyle='--', label='UCL')
            ax_mr.axhline(LCLmr, linestyle='--', label='LCL')
            ax_mr.set_title("Moving Range (MR) chart")
            ax_mr.set_ylabel("MR")
            ax_mr.set_xlabel("Observation (i→i+1)")
            ax_mr.legend()
            st.pyplot(fig_mr)

    else:
        # X-bar & R chart
        if subgroup_means is None or subgroup_ranges is None:
            st.warning("Subgroup information not available – X-bar & R chart cannot be drawn.")
        else:
            xbar_i = np.asarray(subgroup_means, dtype=float)
            R_i = np.asarray(subgroup_ranges, dtype=float)

            xbar_bar = np.mean(xbar_i)
            R_bar = np.mean(R_i)

            if n in control_constants:
                const = control_constants[n]
                A2 = const["A2"]
                D3 = const["D3"]
                D4 = const["D4"]

                x_UCL = xbar_bar + A2 * R_bar
                x_LCL = xbar_bar - A2 * R_bar
                r_UCL = D4 * R_bar
                r_LCL = D3 * R_bar
            else:
                st.info(
                    f"Subgroup size n={n} není v tabulce konstant pro X-bar & R. "
                    "X-bar chart má meze ±3·σ/√n; R chart je bez mezí."
                )
                x_UCL = xbar_bar + 3 * stdev_within / math.sqrt(n)
                x_LCL = xbar_bar - 3 * stdev_within / math.sqrt(n)
                r_UCL = None
                r_LCL = None

            x_values = xbar_i
            x_center = xbar_bar
            r_values = R_i
            r_center = R_bar

            # X-bar chart – použij čas podskupiny, pokud je k dispozici
            fig_xb, ax_xb = plt.subplots()
            if time_subgroups is not None and len(time_subgroups) == len(xbar_i):
                x_axis = time_subgroups
                ax_xb.set_xlabel("Time (subgroup)")
            else:
                x_axis = range(len(xbar_i))
                ax_xb.set_xlabel("Subgroup index")

            ax_xb.plot(x_axis, xbar_i, marker='o', linestyle='-')
            ax_xb.axhline(xbar_bar, linestyle='-', label='CL (mean of subgroup means)')
            ax_xb.axhline(x_UCL, linestyle='--', label='UCL')
            ax_xb.axhline(x_LCL, linestyle='--', label='LCL')
            ax_xb.set_title(f"X-bar chart (n = {n})")
            ax_xb.set_ylabel("Subgroup mean")
            ax_xb.legend()
            st.pyplot(fig_xb)

            # R chart – stejná osa X jako X-bar
            fig_r, ax_r = plt.subplots()
            ax_r.plot(x_axis, R_i, marker='o', linestyle='-')
            ax_r.axhline(R_bar, linestyle='-', label='CL (mean range)')
            if r_UCL is not None and r_LCL is not None:
                ax_r.axhline(r_UCL, linestyle='--', label='UCL')
                ax_r.axhline(r_LCL, linestyle='--', label='LCL')
            ax_r.set_title(f"R chart (n = {n})")
            ax_r.set_ylabel("Range")
            ax_r.legend()
            st.pyplot(fig_r)

    # ==========================
    # EWMA / CUSUM doplňkové grafy
    # ==========================
    st.subheader("Additional control charts (EWMA / CUSUM)")

    chart_base_type = "Individuals" if n == 1 else "Subgroup means"
    st.caption(f"Base for EWMA/CUSUM: {chart_base_type}.")

    if n == 1:
        series_for_advanced = all_values
        x_time_for_advanced = time_values_individuals
    else:
        series_for_advanced = subgroup_means if subgroup_means is not None else all_values
        x_time_for_advanced = time_subgroups

    ewma_cusum_option = st.selectbox(
        "Show additional charts",
        ["None", "EWMA", "CUSUM", "EWMA & CUSUM"],
        index=0
    )

    if ewma_cusum_option in ("EWMA", "EWMA & CUSUM"):
        lam = st.slider("EWMA λ (smoothing factor)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        z, ewma_UCL, ewma_LCL = compute_ewma(series_for_advanced, stdev_within, lam=lam, L=3)
        if z is not None:
            fig_ew, ax_ew = plt.subplots()
            if x_time_for_advanced is not None and len(x_time_for_advanced) == len(series_for_advanced):
                x_axis_adv = x_time_for_advanced
                ax_ew.set_xlabel("Time")
            else:
                x_axis_adv = range(len(series_for_advanced))
                ax_ew.set_xlabel("Index")

            ax_ew.plot(x_axis_adv, series_for_advanced, marker='o', linestyle=':', label='Data')
            ax_ew.plot(x_axis_adv, z, marker='o', linestyle='-', label='EWMA')
            ax_ew.axhline(np.mean(series_for_advanced), linestyle='-', label='CL (mean)')
            ax_ew.axhline(ewma_UCL, linestyle='--', label='UCL')
            ax_ew.axhline(ewma_LCL, linestyle='--', label='LCL')
            ax_ew.set_title("EWMA chart")
            ax_ew.set_ylabel("Value")
            ax_ew.legend()
            st.pyplot(fig_ew)

    if ewma_cusum_option in ("CUSUM", "EWMA & CUSUM"):
        c_plus, c_minus, h_pos, h_neg = compute_cusum(series_for_advanced, stdev_within)
        if c_plus is not None:
            fig_cu, ax_cu = plt.subplots()
            x_axis_cu = range(len(series_for_advanced))
            ax_cu.plot(x_axis_cu, c_plus, marker='o', linestyle='-', label='C+')
            ax_cu.plot(x_axis_cu, c_minus, marker='o', linestyle='-', label='C-')
            ax_cu.axhline(h_pos, linestyle='--', label='+h')
            ax_cu.axhline(h_neg, linestyle='--', label='-h')
            ax_cu.set_title("CUSUM chart")
            ax_cu.set_xlabel("Index")
            ax_cu.set_ylabel("CUSUM")
            ax_cu.legend()
            st.pyplot(fig_cu)

    # ==========================
    # Detekce odlehlých podskupin / bodů
    # ==========================
    st.subheader("Outlier / out-of-control detection")

    outlier_records = []

    # 1) Body mimo kontrolní meze X / X-bar
    if x_values is not None and x_UCL is not None and x_LCL is not None:
        x_array = np.asarray(x_values, dtype=float)
        for i, v in enumerate(x_array):
            if v > x_UCL or v < x_LCL:
                outlier_records.append({
                    "Type": "Control limit (X / Xbar)",
                    "Index": i,
                    "Value": v,
                    "UCL": x_UCL,
                    "LCL": x_LCL,
                })

    # 2) Body mimo kontrolní meze MR / R
    if r_values is not None and r_UCL is not None and r_LCL is not None:
        r_array = np.asarray(r_values, dtype=float)
        for i, v in enumerate(r_array):
            if v > r_UCL or v < r_LCL:
                outlier_records.append({
                    "Type": "Control limit (R / MR)",
                    "Index": i,
                    "Value": v,
                    "UCL": r_UCL,
                    "LCL": r_LCL,
                })

    # 3) Hodnoty mimo specifikaci (LSL/USL)
    all_values_arr = np.asarray(all_values, dtype=float)
    for i, v in enumerate(all_values_arr):
        if v < LSL or v > USL:
            outlier_records.append({
                "Type": "Out of spec (LSL/USL)",
                "Index": i,
                "Value": v,
                "UCL": USL,
                "LCL": LSL,
            })

    if len(outlier_records) == 0:
        st.success("No obvious out-of-control or out-of-spec points detected (simple rules).")
    else:
        st.warning("Some points / subgroups violate control limits or specification limits.")
        outlier_df = pd.DataFrame(outlier_records)
        st.dataframe(outlier_df)

    # ==========================
    # Rozhodnutí
    # ==========================
    st.header('Decision')
    if cust_cpk <= Cpk:
        st.success(f'### ✅ Process is capable Cpk {Cpk:.2f} ✅')
    else:
        st.warning(f'### 🚨 Process is not capable Cpk {Cpk:.2f} 🚨')
