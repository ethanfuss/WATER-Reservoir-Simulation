"""
core/rock_fluid.py
Handles Dykstra-Parsons Permeability Variation (V) and Relative Permeability
Normalization & Power-Law Regression (a, b).

Scan References:
- Scan Page 2–3: "Calculation of Permeability Variation" & "Calculation of Fractional Flow"
- Scan Page 15 (Table 3, Steps 3 & 4): Interactive options to compute or enter V and (a, b)
- Scan Page 18 (Table 5, table5.jpg): Handwritten Swn formula and relative permeability table
- Scan Page 19 (Table 6, table6.jpg): Core permeability layers
- Scan Page 26 (Figure 1A): Subroutines KVA, SWN, LIN1
- Scan Page 34 (Figure 6, fig6.jpg): Permeability Ratio (kro/krw) vs. Normalized Saturation (Swn)
- Scan Page 41 (Appendix A, Eqs. 1, 2, 3): V, kro/krw = a * Swn^(-b), and fw
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def calculate_permeability_variation(core_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates Dykstra-Parsons permeability variation V from core layer permeabilities
    using Bixler's log-normal formulation (Subroutine KVA).

    Scan References:
    - Scan Page 3: "In this program, V can be either entered directly or estimated
      using the formulation of Bixler by entering the permeability of each layer."
    - Scan Page 26 (Figure 1A): SUBR: KVA
    - Scan Page 41 (Appendix A, Eq. 1):
      V = (k50 - k84.1) / k50 = (10^(mean_log_k) - 10^(mean_log_k - sigma)) / 10^(mean_log_k)
    """
    df = pd.DataFrame(core_rows).copy()
    df = df[df["k_md"] > 0].sort_values(by="k_md", ascending=False).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two positive permeability layers are required to estimate V.")

    df["Layer_Rank"] = np.arange(1, len(df) + 1)
    df["log10_k"] = np.log10(df["k_md"].astype(float))
    if "Thickness_ft" not in df.columns:
        df["Thickness_ft"] = 1.0

    # Cumulative probability (% of samples / thickness with larger permeability) for log-normal plot (Scan Page 3)
    total_h = df["Thickness_ft"].sum()
    df["Cum_Thickness_ft"] = df["Thickness_ft"].cumsum()
    df["Pct_Greater"] = ((df["Cum_Thickness_ft"] - 0.5 * df["Thickness_ft"]) / total_h) * 100.0

    # Scan Page 41, Eq. 1 (Subroutine KVA uses sample standard deviation of log10(k))
    mean_log_k = float(df["log10_k"].mean())
    sigma_log_k = float(df["log10_k"].std(ddof=1))

    k50 = 10.0 ** mean_log_k
    k84_1 = 10.0 ** (mean_log_k - sigma_log_k)
    v_bixler = (k50 - k84_1) / k50

    return {
        "V": float(v_bixler),
        "k50": float(k50),
        "k84_1": float(k84_1),
        "mean_log_k": mean_log_k,
        "sigma_log_k": sigma_log_k,
        "layer_df": df,
    }


def calculate_normalized_saturation(sw: float, swi: float, sor: float) -> float:
    """
    Calculates Normalized Water Saturation (Swn) according to Subroutine SWN.

    Scan References:
    - Scan Page 18 (Table 5, table5.jpg): Handwritten formula at bottom of Table 5:
      Swn = (Sw - Swi) / (1 - Sw - Sor)
      Verified against handwritten column values in table5.jpg:
      Sw=0.30 -> 0.5, Sw=0.40 -> 1.0, Sw=0.45 -> 1.4, Sw=0.50 -> 2.0,
      Sw=0.55 -> 3.0, Sw=0.60 -> 5.0, Sw=0.65 -> 11.0.
    - Scan Page 26 (Figure 1A): SUBR: SWN
    - Scan Page 41 (Appendix A): Required for Eqs. 2 and 4.
    """
    denom = 1.0 - sw - sor
    if denom <= 1e-12 or sw <= swi:
        return 0.0
    return float((sw - swi) / denom)


def fit_rel_perm_power_law(
    rel_perm_rows: List[Dict[str, Any]],
    swi: float,
    sor: float,
) -> Dict[str, Any]:
    """
    Fits the power-law correlation kro/krw = a * Swn^(-b) via log-linear least squares
    regression (Subroutine LIN1).

    Scan References:
    - Scan Page 3: "To simplify the calculations, a simple correlation of kro/krw vs.
      normalized water saturation, Swn, was chosen (Eq. 2 in Appendix A)."
    - Scan Page 26 (Figure 1A): SUBR: LIN1
    - Scan Page 34 (Figure 6, fig6.jpg): Log-log plot of kro/krw vs. Swn
    - Scan Page 41 (Appendix A, Eq. 2): kro / krw = a * Swn^(-b)
    """
    swn_vals = []
    ratio_vals = []

    for row in rel_perm_rows:
        sw = float(row["Sw"])
        kro = float(row["kro"])
        krw = float(row["krw"])
        include = bool(row.get("fit_include", True))
        if include and kro > 0 and krw > 0 and (sw > swi) and (1.0 - sw - sor > 1e-9):
            swn = calculate_normalized_saturation(sw, swi, sor)
            swn_vals.append(swn)
            ratio_vals.append(kro / krw)

    # Fallback: if fewer than 2 points marked fit_include, use all valid interior points
    if len(swn_vals) < 2:
        swn_vals = []
        ratio_vals = []
        for row in rel_perm_rows:
            sw = float(row["Sw"])
            kro = float(row["kro"])
            krw = float(row["krw"])
            if kro > 0 and krw > 0 and (sw > swi) and (1.0 - sw - sor > 1e-9):
                swn_vals.append(calculate_normalized_saturation(sw, swi, sor))
                ratio_vals.append(kro / krw)

    if len(swn_vals) < 2:
        raise ValueError("At least two valid relative permeability points (kro > 0, krw > 0) are required.")

    log_x = np.log(np.array(swn_vals, dtype=float))
    log_y = np.log(np.array(ratio_vals, dtype=float))

    # Linear regression: ln(kro/krw) = ln(a) - b * ln(Swn)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    b = float(-slope)
    a = float(np.exp(intercept))

    # Calculate R^2 of log-log fit
    y_pred = intercept + slope * log_x
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "n_points_used": len(swn_vals),
    }


def build_rel_perm_table(
    rel_perm_rows: List[Dict[str, Any]],
    swi: float,
    sor: float,
    mu_o: float,
    mu_w: float,
    a: float,
    b: float,
) -> pd.DataFrame:
    """
    Builds the complete intermediate Relative Permeability & Fractional Flow table
    matching Table 5 (Scan Page 18, table5.jpg) and Figure 6 (Scan Page 34, fig6.jpg).

    Columns generated:
    - Sw (fraction)
    - Swn (normalized saturation, handwritten col in table5.jpg)
    - kro (fraction)
    - krw (fraction)
    - kro/krw Actual (handwritten col in table5.jpg & circles in fig6.jpg)
    - kro/krw Correlated (a * Swn^(-b), squares/triangles in fig6.jpg)
    - fw Actual (Table 5 col 4, Eq. 3 using actual kro/krw)
    - fw Correlated (Eq. 3 using correlated kro/krw)
    """
    records = []
    for row in rel_perm_rows:
        sw = float(row["Sw"])
        kro = float(row["kro"])
        krw = float(row["krw"])
        include = bool(row.get("fit_include", True))

        if sw <= swi or krw <= 0.0:
            swn = np.nan
            ratio_actual = np.nan
            ratio_corr = np.nan
            fw_actual = 0.0
            fw_corr = 0.0
        elif (1.0 - sw - sor) <= 1e-9 or kro <= 0.0:
            swn = np.nan
            ratio_actual = 0.0 if krw > 0 else np.nan
            ratio_corr = 0.0
            fw_actual = 1.0
            fw_corr = 1.0
        else:
            swn = calculate_normalized_saturation(sw, swi, sor)
            ratio_actual = kro / krw
            ratio_corr = a * (swn ** (-b))
            # Scan Page 41, Eq. 3: fw = 1 / (1 + (kro/krw) * (mu_w / mu_o))
            fw_actual = 1.0 / (1.0 + ratio_actual * (mu_w / mu_o))
            fw_corr = 1.0 / (1.0 + ratio_corr * (mu_w / mu_o))

        records.append({
            "Sw": sw,
            "Swn": swn,
            "kro": kro,
            "krw": krw,
            "kro_krw_actual": ratio_actual,
            "kro_krw_corr": ratio_corr,
            "fw_actual": fw_actual,
            "fw_corr": fw_corr,
            "fit_include": include,
        })

    return pd.DataFrame(records)