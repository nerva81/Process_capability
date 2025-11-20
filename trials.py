import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('MSA_type2.csv', sep=';')

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

if use_interaction:
    # ===== Model S interakcí (obecný náhodný model) =====
    MS_part = MS_part_int
    MS_op = MS_op_int
    MS_repeat = MS_repeat_int

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

    # Interakce se považuje za nulovou
    var_repeat = MS_repeat
    var_op_part = 0.0
    # Vzorce podle MSA bez interakce:
    # σ²_O = (MS_O - MS_E) / (b * n)  ; σ²_P = (MS_P - MS_E) / (a * n)
    var_operator = max((MS_op - MS_repeat) / (s * r), 0)
    var_part = max((MS_part - MS_repeat) / (t * r), 0)

# 3) Ostatní složky
var_reprod = var_operator + var_op_part      # AV (reproducibility)
var_gage = var_repeat + var_reprod          # GRR
var_total = var_gage + var_part             # Total

sd_repeat = math.sqrt(var_repeat) if var_repeat > 0 else 0.0
sd_reprod = math.sqrt(var_reprod) if var_reprod > 0 else 0.0
sd_gage = math.sqrt(var_gage) if var_gage > 0 else 0.0
sd_part = math.sqrt(var_part) if var_part > 0 else 0.0
sd_total = math.sqrt(var_total) if var_total > 0 else 0.0

# 4) ndc
ndc = 1.41 * sd_part / sd_gage if sd_gage > 0 else np.nan

# 5) Přehledný výpis (f-string)
overview = (
    f"parts = {parts}\n"
    f"ops = {ops}\n"
    f"n_parts = {n_parts}\n"
    f"n_ops = {n_ops}\n"
    f"cell_counts =\n{cell_counts}\n"
    f"n_rep_mean = {n_rep_mean:.3f}\n"
    f"is_balanced = {is_balanced}\n"
    f"r (repeats per cell) = {r}\n"
    "\n"
    "=== ANOVA s interakcí ===\n"
    f"{anova_int}\n"
    f"MS_int = {MS_int:.5f}, MS_repeat_int = {MS_repeat_int:.5f}, p_int = {p_int:.3f}\n"
    f"use_interaction = {use_interaction}\n"
    "\n"
    "=== Použitý ANOVA model (po rozhodnutí o interakci) ===\n"
    f"MS_part = {MS_part:.5f}\n"
    f"MS_op = {MS_op:.5f}\n"
    f"MS_repeat = {MS_repeat:.5f}\n"
    "\n"
    "=== Variance components ===\n"
    f"var_repeat (EV)       = {var_repeat:.5f}\n"
    f"var_op_part (OP×Part) = {var_op_part:.5f}\n"
    f"var_operator          = {var_operator:.5f}\n"
    f"var_part              = {var_part:.5f}\n"
    f"var_reprod (AV)       = {var_reprod:.5f}\n"
    f"var_gage (GRR)        = {var_gage:.5f}\n"
    f"var_total             = {var_total:.5f}\n"
    "\n"
    "=== Standard deviations ===\n"
    f"sd_repeat (EV)   = {sd_repeat:.5f}\n"
    f"sd_reprod (AV)   = {sd_reprod:.5f}\n"
    f"sd_gage (GRR)    = {sd_gage:.5f}\n"
    f"sd_part (PV)     = {sd_part:.5f}\n"
    f"sd_total (TV)    = {sd_total:.5f}\n"
    "\n"
    f"ndc = {ndc:.3f}\n"
)

print(overview)
