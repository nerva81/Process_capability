# backend/msa_core.py

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# d2 konstanty pro Xbar-R metodu (subgroup size)
D2_CONSTANT = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
}


# ==========================
# Cohen & Fleiss kappa (bez sklearn)
# ==========================

def cohen_kappa(y_true, y_pred) -> float:
    """
    Cohenova kappa bez sklearn, podporuje multi-class.
    """
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    N = len(y_true)
    if N == 0:
        return np.nan

    cm = pd.crosstab(y_true, y_pred)
    po = np.trace(cm.values) / N

    row_marg = cm.sum(axis=1).values
    col_marg = cm.sum(axis=0).values
    pe = np.sum(row_marg * col_marg) / (N ** 2)

    if np.isclose(1 - pe, 0):
        return np.nan

    kappa = (po - pe) / (1 - pe)
    return kappa


def fleiss_kappa_from_long(part, rater, rating) -> float:
    """
    Fleissova kappa z long-form dat:
    každý řádek = (part, rater, rating).
    Funguje i pro nevyvážený design přibližně.
    """
    df = pd.DataFrame({"Part": part, "Rating": rating}).dropna()
    if df.empty:
        return np.nan

    df["Rating"] = df["Rating"].astype(str)
    parts = sorted(df["Part"].unique())
    cats = sorted(df["Rating"].unique())

    counts = pd.crosstab(df["Part"], df["Rating"]).reindex(
        index=parts, columns=cats, fill_value=0
    )

    N = counts.shape[0]
    n_i = counts.sum(axis=1).values
    n_bar = n_i.mean()

    if N == 0 or n_bar == 0:
        return np.nan

    P_i = []
    for i in range(N):
        n_ij = counts.iloc[i, :].values
        n = n_i[i]
        if n <= 1:
            P_i.append(0.0)
        else:
            P_i.append((np.sum(n_ij * (n_ij - 1))) / (n * (n - 1)))
    P_bar = np.mean(P_i)

    n_ij_sum = counts.sum(axis=0).values
    p_j = n_ij_sum / (N * n_bar)
    P_e = np.sum(p_j ** 2)

    if np.isclose(1 - P_e, 0):
        return np.nan

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


# ==========================
# Core výpočty pro Type 2 – ANOVA & Xbar-R
# ==========================

def compute_grr_anova_core(data: pd.DataFrame, tol: float | None = None) -> dict:
    """
    Core výpočet pro Type 2 Gage R&R – ANOVA metoda.
    Očekává sloupce: Part, Operator, Measurement (numeric).

    Vrací dict s komponentami (EV, Reprod, GRR, PV, TV, %StudyVar, %Tolerance, ndc).
    """
    parts = data["Part"].unique()
    ops = data["Operator"].unique()
    n_parts = len(parts)
    n_ops = len(ops)

    cell_counts = data.groupby(["Part", "Operator"]).size()
    n_rep_mean = cell_counts.mean()
    is_balanced = (cell_counts.nunique() == 1)
    r = cell_counts.iloc[0] if is_balanced else n_rep_mean

    # ANOVA model
    model = smf.ols("Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)", data=data).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    anova_tbl["MS"] = anova_tbl["sum_sq"] / anova_tbl["df"]

    MS_part = anova_tbl.loc["C(Part)", "MS"]
    MS_op = anova_tbl.loc["C(Operator)", "MS"]
    MS_int = anova_tbl.loc["C(Part):C(Operator)", "MS"]
    MS_repeat = anova_tbl.loc["Residual", "MS"]

    t = n_ops
    s = n_parts

    # Variance components
    var_repeat = MS_repeat
    var_op_part = max((MS_int - MS_repeat) / r, 0)
    var_operator = max((MS_op - MS_int) / (r * s), 0)
    var_part = max((MS_part - MS_int) / (r * t), 0)

    var_reprod = var_operator + var_op_part
    var_gage = var_repeat + var_reprod
    var_total = var_gage + var_part

    sd_repeat = math.sqrt(var_repeat) if var_repeat > 0 else 0.0
    sd_reprod = math.sqrt(var_reprod) if var_reprod > 0 else 0.0
    sd_gage = math.sqrt(var_gage) if var_gage > 0 else 0.0
    sd_part = math.sqrt(var_part) if var_part > 0 else 0.0
    sd_total = math.sqrt(var_total) if var_total > 0 else 0.0

    # ndc
    ndc = 1.41 * sd_part / sd_gage if sd_gage > 0 else np.nan

    def pct_sv(sd):
        return 100 * sd / sd_total if sd_total > 0 else np.nan

    def pct_tol(sd):
        if tol is None or tol <= 0:
            return np.nan
        return 100 * (6 * sd) / tol

    return {
        "method": "ANOVA",
        "EV": sd_repeat,
        "Reprod": sd_reprod,
        "GRR": sd_gage,
        "PV": sd_part,
        "TV": sd_total,
        "%SV_EV": pct_sv(sd_repeat),
        "%SV_Reprod": pct_sv(sd_reprod),
        "%SV_GRR": pct_sv(sd_gage),
        "%SV_PV": pct_sv(sd_part),
        "%SV_TV": 100.0,
        "%Tol_EV": pct_tol(sd_repeat),
        "%Tol_Reprod": pct_tol(sd_reprod),
        "%Tol_GRR": pct_tol(sd_gage),
        "%Tol_PV": pct_tol(sd_part),
        "%Tol_TV": pct_tol(sd_total),
        "ndc": ndc,
    }


def compute_grr_xbar_core(data: pd.DataFrame, tol: float | None = None) -> dict:
    """
    Core výpočet pro Type 2 Gage R&R – X̄–R metoda.
    Očekává sloupce: Part, Operator, Measurement (numeric).
    """
    subgroup_sizes = data.groupby(["Part", "Operator"]).size().unique()
    if len(subgroup_sizes) != 1:
        raise ValueError("X̄–R requires balanced design (same repeats per Part × Operator).")

    n = int(subgroup_sizes[0])
    if n not in D2_CONSTANT:
        raise ValueError(f"No d2 constant for subgroup size n={n}. Supported: {list(D2_CONSTANT.keys())}.")

    d2 = D2_CONSTANT[n]
    parts = data["Part"].unique()
    ops = data["Operator"].unique()
    k_parts = len(parts)
    m_ops = len(ops)

    ranges = data.groupby(["Part", "Operator"])["Measurement"].apply(lambda x: x.max() - x.min())
    Rbar = ranges.mean()
    EV = Rbar / d2

    op_means = data.groupby("Operator")["Measurement"].mean()
    overall_mean = data["Measurement"].mean()

    AV_raw = (k_parts * ((op_means - overall_mean) ** 2).sum()) / (m_ops - 1)
    AV = math.sqrt(max(AV_raw - EV**2, 0))

    part_means = data.groupby("Part")["Measurement"].mean()
    PV_raw = (m_ops * ((part_means - overall_mean) ** 2).sum()) / (k_parts - 1)
    PV = math.sqrt(max(PV_raw, 0))

    GRR = math.sqrt(EV**2 + AV**2)
    TV = math.sqrt(EV**2 + AV**2 + PV**2)

    ndc = 1.41 * PV / GRR if GRR > 0 else np.nan

    def pct_sv(sd):
        return 100 * sd / TV if TV > 0 else np.nan

    def pct_tol(sd):
        if tol is None or tol <= 0:
            return np.nan
        return 100 * (6 * sd) / tol

    return {
        "method": "X̄–R",
        "EV": EV,
        "Reprod": AV,
        "GRR": GRR,
        "PV": PV,
        "TV": TV,
        "%SV_EV": pct_sv(EV),
        "%SV_Reprod": pct_sv(AV),
        "%SV_GRR": pct_sv(GRR),
        "%SV_PV": pct_sv(PV),
        "%SV_TV": 100.0,
        "%Tol_EV": pct_tol(EV),
        "%Tol_Reprod": pct_tol(AV),
        "%Tol_GRR": pct_tol(GRR),
        "%Tol_PV": pct_tol(PV),
        "%Tol_TV": pct_tol(TV),
        "ndc": ndc,
    }
