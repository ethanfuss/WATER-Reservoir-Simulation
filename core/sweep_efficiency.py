"""
core/sweep_efficiency.py
Calculates Mobility Ratio (M) before and after breakthrough, Areal Sweep Efficiency (Ea),
and Dykstra-Parsons Vertical Coverage (C).

Scan References:
- Scan Page 4–5: "Calculation of Areal Sweep Efficiency" & "Calculation of Vertical Coverage"
- Scan Page 13 (Table 1): Pattern regression coefficients for 5-Spot, DLD, and SLD
- Scan Page 14 (Table 2): Coverage comparison table (Ref. 3 vs Eq. 16)
- Scan Page 28 (Figure 1C): Flowchart for Areal Sweep Efficiency Subroutine EA
- Scan Page 29 (Figure 1D): Flowchart for Vertical Coverage Subroutine C
- Scan Page 31 (Figure 3): Actual vs. Correlated Areal Sweep Efficiency
- Scan Page 32–33 (Figures 4 & 5): Coverage vs. V, M, and Y Correlation Parameter
- Scan Pages 42–44 (Appendix A, Eqs. 11–21): Equations for M, Ea, Y, and C
"""

from typing import List, Dict, Any
import numpy as np
from .presets import PATTERN_COEFFICIENTS, COVERAGE_CONSTANTS


def interpolate_krw_at_sw(
    sw_target: float,
    rel_perm_rows: List[Dict[str, Any]],
    swi: float,
    sor: float,
    kro_swi: float,
) -> float:
    """
    Linearly interpolates relative permeability to water krw(Sw_avg) from the user's
    relative permeability table (Table 5 / Table 8).
    """
    sw_list = [float(r["Sw"]) for r in rel_perm_rows]
    krw_list = [float(r["krw"]) for r in rel_perm_rows]
    order = np.argsort(sw_list)
    sw_sorted = np.array(sw_list)[order]
    krw_sorted = np.array(krw_list)[order]
    return float(np.interp(sw_target, sw_sorted, krw_sorted))


def calculate_mobility_ratio(
    sw_avg: float,
    fw_at_sw_avg: float,
    is_after_breakthrough: bool,
    rel_perm_rows: List[Dict[str, Any]],
    swi: float,
    sor: float,
    kro_swi: float,
    mu_o: float,
    mu_w: float,
    after_bt_method: str = "endpoint_krw",
) -> Dict[str, float]:
    """
    Calculates Mobility Ratio (M) with distinct layers before and after breakthrough.

    Scan References:
    - Scan Page 4: "This parameter is a function of the mobility of the injected fluid
      to the mobility of reservoir oil, M" -> M = (krw(Sw_avg)/mu_w) / (kro(Soi)/mu_o).
    - Scan Page 20 (Table 7, table7.jpg):
      * Before breakthrough (WOR <= 2.0, fw < fwbt): Sw_avg = Sw_avg_bt = 0.563 -> M = 0.80.
      * After breakthrough (WOR >= 5.0, fw >= fwbt): Sw_avg increases with WOR ->
        M increases from 0.84 (WOR=5) -> 0.99 (WOR=10) -> 1.18 (WOR=25) -> 1.28 (WOR=50) -> 1.35 (WOR=100).
    - Scan Pages 42–43 (Appendix A, Eqs. 11–13):
      * Up to breakthrough (Eq. 11): M = (krw(Sw_avg) * mu_o) / (kro(Soi) * mu_w)
      * After breakthrough (Eq. 12): M_total = (1 / fw(Sw_avg)) * [(krw(Sw_avg) * mu_o) / (kro(Soi) * mu_w)]
    """
    krw_sw_avg = interpolate_krw_at_sw(sw_avg, rel_perm_rows, swi, sor, kro_swi)
    m_endpoint = (krw_sw_avg * mu_o) / (kro_swi * mu_w)

    if not is_after_breakthrough:
        # Before breakthrough (Eq. 11, Scan Page 42)
        m_val = m_endpoint
    else:
        # After breakthrough: supports both Table 7's krw(Sw_avg) method and Eq. 12's 1/fw(Sw_avg) factor
        if after_bt_method == "eq12_total_mobility" and fw_at_sw_avg > 0:
            m_val = m_endpoint / fw_at_sw_avg
        else:
            m_val = m_endpoint

    return {
        "M": float(m_val),
        "M_endpoint": float(m_endpoint),
        "M_reciprocal_eq13": float(1.0 / m_val) if m_val > 0 else 0.0,
        "krw_sw_avg": float(krw_sw_avg),
    }


def calculate_areal_sweep(
    m: float,
    fw: float,
    pattern: str = "5-Spot",
) -> Dict[str, float]:
    """
    Calculates Areal Sweep Efficiency (Ea) using Fassihi's nonlinear regression correlation
    of the Dyes, Caudle, & Erickson experimental graphs (Subroutine EA).

    Scan References:
    - Scan Page 4: (1 - Ea) / Ea = (a1 * ln(M + a2) + a3) * fw + a4 * ln(M + a5) + a6
    - Scan Page 13 (Table 1): Pattern coefficients a1..a6 for 5-Spot, Direct Line, Staggered Line
    - Scan Page 28 (Figure 1C): Flowchart for Subroutine EA
    - Scan Page 43 (Appendix A, Eqs. 14–15):
      Ea = 1 / (1 + Y_areal)
      Y_areal = (a1 * ln(a2 + M) + a3) * fw + a4 * ln(a5 + M) + a6
    """
    coeffs = PATTERN_COEFFICIENTS.get(pattern, PATTERN_COEFFICIENTS["5-Spot"])
    a1, a2, a3, a4, a5, a6 = (
        coeffs["a1"],
        coeffs["a2"],
        coeffs["a3"],
        coeffs["a4"],
        coeffs["a5"],
        coeffs["a6"],
    )

    # Guard arguments inside natural logarithms for extreme mobility ratios
    arg1 = max(m + a2, 1e-6)
    arg2 = max(m + a5, 1e-6)

    # Eq. 15 (Scan Page 43)
    y_areal = (a1 * np.log(arg1) + a3) * fw + a4 * np.log(arg2) + a6

    # Flowchart Figure 1C (Scan Page 28): If Y_areal <= 0, Ea = 1.0
    if y_areal <= 0.0:
        ea = 1.0
    else:
        # Eq. 14 (Scan Page 43): Ea = 1 / (1 + Y_areal)
        ea = 1.0 / (1.0 + y_areal)

    ea = float(np.clip(ea, 0.0, 1.0))
    return {
        "Ea": ea,
        "Y_areal": float(max(y_areal, 0.0)),
    }


def calculate_coverage_parameter_y(wor: float, m: float, v: float) -> float:
    """
    Calculates the de Souza & Brigham vertical coverage grouping parameter Y.

    Scan References:
    - Scan Page 4 & Scan Page 43 (Appendix A, Eq. 17) & SPE 13945 Page 604 (Eq. 1):
      Y = [(WOR + 0.4) * (18.948 - 2.499 * V)] / [(M + 1.137 - 0.8094 * V) * 10^f(V)]
      where f(V) = -0.6891 + 0.9735 * V + 1.6453 * V^2
    """
    f_v = -0.6891 + 0.9735 * v + 1.6453 * (v ** 2)
    numerator = (wor + 0.4) * (18.948 - 2.499 * v)
    denominator = (m + 1.137 - 0.8094 * v) * (10.0 ** f_v)
    if denominator <= 1e-12:
        return 0.0
    return float(max(numerator / denominator, 0.0))


def solve_vertical_coverage(wor: float, m: float, v: float, tol: float = 1e-5) -> Dict[str, Any]:
    """
    Solves for Vertical Coverage (C) from the correlation parameter Y using the exact
    4-regime piecewise trial-and-error algorithm in Subroutine C.

    Scan References:
    - Scan Page 5 & Scan Page 14 (Table 2): Validation of C from Eq. 16
    - Scan Page 29 (Figure 1D): Coverage Subroutine Flowchart
    - Scan Pages 43–44 (Appendix A, Eqs. 16–21):
      Eq. 16: Y = a1 * C^a2 * (1 - C)^a3
        where a1 = 3.334088568, a2 = 0.7737348199, a3 = -1.225859406
      Regime 1 (Eq. 18): Y < 0.5          -> C = 0
      Regime 2 (Eq. 19): 0.5 <= Y <= 2.6  -> C_{i+1} = [Y / (a1 * (1 - C_i)^a3)]^(1 / a2), init C0 = 0.3
      Regime 3 (Eq. 20): 2.6 < Y <= 5.0   -> C = (ln(Y) + 0.60414) / 4.2569
      Regime 4 (Eq. 21): Y > 5.0          -> C_{i+1} = 1 - [Y / (a1 * C_i^a2)]^(1 / a3), init C0 = 0.65
    """
    y = calculate_coverage_parameter_y(wor, m, v)
    a1 = COVERAGE_CONSTANTS["a1"]
    a2 = COVERAGE_CONSTANTS["a2"]
    a3 = COVERAGE_CONSTANTS["a3"]

    # Note: In Table 2 (Scan Page 14), Y = 0.41 gives C = 0.06 and Y = 0.17 gives C = 0.02,
    # whereas Flowchart Fig. 1D (Scan Page 29) has a fast cutoff for very small Y.
    # We compute both the exact fixed-point solution of Eq. 16 and record the regime.
    if y <= 0.05:
        c_val = 0.0
        regime = "Regime 1 (Eq. 18: Very Small Y -> C ≈ 0)"
    elif y <= 2.6:
        regime = "Regime 2 (Eq. 19: 0.5 <= Y <= 2.6 Fixed-Point Iteration from C0=0.30)" if y >= 0.5 else "Regime 1/2 Boundary (Eq. 19 Fixed-Point Iteration)"
        c_curr = 0.30
        for _ in range(500):
            c_next = (y / (a1 * ((1.0 - c_curr) ** a3))) ** (1.0 / a2)
            c_next = float(np.clip(c_next, 1e-6, 0.999999))
            if abs(c_next - c_curr) < tol:
                c_curr = c_next
                break
            c_curr = 0.5 * c_curr + 0.5 * c_next
        c_val = c_curr
    elif y <= 5.0:
        # Eq. 20 (Scan Page 44)
        regime = "Regime 3 (Eq. 20: 2.6 < Y <= 5.0 Logarithmic Formula)"
        c_val = (np.log(y) + 0.60414) / 4.2569
    else:
        # Eq. 21 (Scan Page 44)
        regime = "Regime 4 (Eq. 21: Y > 5.0 Fixed-Point Iteration from C0=0.65)"
        c_curr = 0.65
        for _ in range(500):
            c_next = 1.0 - (y / (a1 * (c_curr ** a2))) ** (1.0 / a3)
            c_next = float(np.clip(c_next, 1e-6, 0.999999))
            if abs(c_next - c_curr) < tol:
                c_curr = c_next
                break
            c_curr = 0.5 * c_curr + 0.5 * c_next
        c_val = c_curr

    c_val = float(np.clip(c_val, 0.0, 1.0))
    return {
        "Y": float(y),
        "C": c_val,
        "regime": regime,
    }