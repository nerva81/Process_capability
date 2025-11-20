# backend/msa_core.py

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# K konstanty pro Xbar-R metodu (subgroup size)
K1_CONSTANT = {
    2: 0.8862,
    3: 0.5908,
}


K2_CONSTANT = {
    2: 0.7071,
    3: 0.5231,
}


K3_CONSTANT = {
    2: 0.7071,
    3: 0.5231,
    4: 0.4467,
    5: 0.4030,
    6: 0.3742,
    7: 0.3534,
    8: 0.3375,
    9: 0.3249,
    10: 0.3146,
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

    Vrací dict s komponentami (EV, Reprod, GRR, PV, TV, %StudyVar, %Tolerance, ndc)
    + navíc ANOVA tabulky a variance components ve stylu Minitabu.
    """
    parts = data["Part"].unique()
    ops = data["Operator"].unique()
    n_parts = len(parts)
    n_ops = len(ops)

    cell_counts = data.groupby(["Part", "Operator"]).size()
    n_rep_mean = cell_counts.mean()
    is_balanced = (cell_counts.nunique() == 1)
    r = cell_counts.iloc[0] if is_balanced else n_rep_mean  # počet opakování na buňku

    # 1) ANOVA model S interakcí
    model_int = smf.ols("Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)", data=data).fit()
    anova_int = sm.stats.anova_lm(model_int, typ=2)
    anova_int["MS"] = anova_int["sum_sq"] / anova_int["df"]

    MS_part_int = anova_int.loc["C(Part)", "MS"]
    MS_op_int = anova_int.loc["C(Operator)", "MS"]
    MS_int = anova_int.loc["C(Part):C(Operator)", "MS"]
    MS_repeat_int = anova_int.loc["Residual", "MS"]
    p_int = anova_int.loc["C(Part):C(Operator)", "PR(>F)"]

    t = n_ops
    s = n_parts

    # 2) Rozhodnutí: použít model s/bez interakce (MSA + Minitab logika)
    #    - pokud MS(interakce) < MS(error) NEBO p > 0.25 => interakce zanedbatelná, model bez interakce
    use_interaction = not (MS_int < MS_repeat_int or p_int > 0.25)

    # ANOVA tabulka použitého modelu (naplníme níže)
    anova_used = None

    if use_interaction:
        # ===== Model S interakcí (obecný náhodný model) =====
        MS_part = MS_part_int
        MS_op = MS_op_int
        MS_repeat = MS_repeat_int
        anova_used = anova_int

        # Variance components (random Part & Operator)
        var_repeat = MS_repeat
        var_op_part = max((MS_int - MS_repeat) / r, 0)
        var_operator = max((MS_op - MS_int) / (r * s), 0)
        var_part = max((MS_part - MS_int) / (r * t), 0)

    else:
        # ===== Model BEZ interakce (MSA / Minitab při nevýznamné interakci) =====
        model_no_int = smf.ols("Measurement ~ C(Part) + C(Operator)", data=data).fit()
        anova_no_int = sm.stats.anova_lm(model_no_int, typ=2)
        anova_no_int["MS"] = anova_no_int["sum_sq"] / anova_no_int["df"]

        MS_part = anova_no_int.loc["C(Part)", "MS"]
        MS_op = anova_no_int.loc["C(Operator)", "MS"]
        MS_repeat = anova_no_int.loc["Residual", "MS"]
        anova_used = anova_no_int

        # Interakce se považuje za nulovou
        var_repeat = MS_repeat
        var_op_part = 0.0
        # Vzorce podle MSA bez interakce:
        # σ²_O = (MS_O - MS_E) / (b * n)  ; σ²_P = (MS_P - MS_E) / (a * n)
        var_operator = max((MS_op - MS_repeat) / (s * r), 0)
        var_part = max((MS_part - MS_repeat) / (t * r), 0)

    # 3) Ostatní složky (variance + SD)
    var_reprod = var_operator + var_op_part  # AV (reproducibility)
    var_gage = var_repeat + var_reprod       # GRR
    var_total = var_gage + var_part          # Total

    sd_repeat = math.sqrt(var_repeat) if var_repeat > 0 else 0.0
    sd_reprod = math.sqrt(var_reprod) if var_reprod > 0 else 0.0
    sd_gage = math.sqrt(var_gage) if var_gage > 0 else 0.0
    sd_part = math.sqrt(var_part) if var_part > 0 else 0.0
    sd_total = math.sqrt(var_total) if var_total > 0 else 0.0

    # 4) ndc
    ndc = 1.41 * sd_part / sd_gage if sd_gage > 0 else np.nan

    # 5) %Study variation a %Tolerance
    def pct_sv(sd):
        return 100 * sd / sd_total if sd_total > 0 else np.nan

    def pct_tol(sd):
        if tol is None or tol <= 0:
            return np.nan
        return 100 * (6 * sd) / tol

    return {
        "method": "ANOVA",

        # ===== základní SD komponenty (co už používáš) =====
        "EV": sd_repeat,
        "Reprod": sd_reprod,
        "GRR": sd_gage,
        "PV": sd_part,
        "TV": sd_total,

        # % Study variation
        "%SV_EV": pct_sv(sd_repeat),
        "%SV_Reprod": pct_sv(sd_reprod),
        "%SV_GRR": pct_sv(sd_gage),
        "%SV_PV": pct_sv(sd_part),
        "%SV_TV": 100.0,

        # % Tolerance
        "%Tol_EV": pct_tol(sd_repeat),
        "%Tol_Reprod": pct_tol(sd_reprod),
        "%Tol_GRR": pct_tol(sd_gage),
        "%Tol_PV": pct_tol(sd_part),
        "%Tol_TV": pct_tol(sd_total),

        # ndc
        "ndc": ndc,

        # ===== ANOVA tabulky pro frontend =====
        "anova_with_int": anova_int,
        "anova_used": anova_used,
        "use_interaction": use_interaction,

        # ===== variance components (Minitab-style) =====
        "var_EV": var_repeat,
        "var_OPxPart": var_op_part,
        "var_Operator": var_operator,
        "var_Reprod": var_reprod,
        "var_GRR": var_gage,
        "var_PV": var_part,
        "var_TV": var_total,

        # volitelné debug/info klíče navíc (můžeš kdykoliv smazat, pokud je nechceš)
        "MS_int": MS_int,
        "MS_error": MS_repeat,
        "p_int": p_int,
        "n_parts": n_parts,
        "n_ops": n_ops,
        "r": r,
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
    if n not in K1_CONSTANT:
        raise ValueError(f"No k1 constant for subgroup size n={n}. Supported: {list(D2_CONSTANT.keys())}.")

    k1 = K1_CONSTANT[n]
    parts = data["Part"].unique()
    ops = data["Operator"].unique()
    k_parts = len(parts)
    m_ops = len(ops)
    k2 = K2_CONSTANT[m_ops]
    ranges = data.groupby(["Part", "Operator"])["Measurement"].apply(lambda x: x.max() - x.min())
    Rbar = ranges.mean()
    op_means = data.groupby("Operator")["Measurement"].mean()
    Xbar = max(op_means) - min(op_means)
    overall_mean = data["Measurement"].mean()
    EV = Rbar * k1
    AV_raw = (k_parts * ((op_means - overall_mean) ** 2).sum()) / (m_ops - 1)
    AV = math.sqrt((Xbar * k2) ** 2 - (EV ** 2 / (n * k_parts)))
    part_means = data.groupby("Part")["Measurement"].mean()
    parts_R = max(part_means) - min(part_means)
    k3 = K3_CONSTANT[k_parts]
    PV_raw = (m_ops * ((part_means - overall_mean) ** 2).sum()) / (k_parts - 1)
    PV = parts_R * k3

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
