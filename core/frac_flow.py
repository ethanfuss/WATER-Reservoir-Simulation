"""
core/frac_flow.py
Fractional Flow (fw), Derivative (fw'), Welge Tangent Construction at Breakthrough (Subroutine SWBT),
and Post-Breakthrough Saturation Calculations.

Scan References:
- Scan Page 3: "Calculation of Fractional Flow" & "Calculation of f'w"
- Scan Page 26–27 (Figures 1A & 1B): Flowchart for Swbt & Subroutine SLP/FWB
- Scan Page 30 (Figure 2): Chord and tangent geometry (Sw1, Sw2, Slope2, Slope3)
- Scan Page 35 (Figure 7, fig7.jpg): Fractional Flow of Water for Example 1
- Scan Page 38 (Figure 10): Fractional Flow of Water for Example 2
- Scan Page 41–42 (Appendix A, Eqs. 3–10): Mathematical equations for fw, fw', Swbt, Sw_avg
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from .rock_fluid import calculate_normalized_saturation

def calculate_fw_from_ratio(kro_krw: float, mu_o: float, mu_w: float) -> float:
    """
    Calculates fractional flow of water fw given kro/krw and viscosities.
    Tracks to Scan Page 41, Appendix A, Eq. 3:
    fw = 1 / (1 + (kro / krw) * (mu_w / mu_o))
    """
    if kro_krw <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + kro_krw * (mu_w / mu_o)))


def calculate_fw_from_sw(
    sw: float,
    swi: float,
    sor: float,
    a: float,
    b: float,
    mu_o: float,
    mu_w: float,
) -> float:
    """
    Calculates fractional flow fw(Sw) using Eqs. 2 and 3 (Scan Page 41).
    """
    if sw <= swi:
        return 0.0
    if sw >= (1.0 - sor):
        return 1.0
    swn = calculate_normalized_saturation(sw, swi, sor)
    if swn <= 0.0:
        return 0.0
    kro_krw = a * (swn ** (-b))
    return calculate_fw_from_ratio(kro_krw, mu_o, mu_w)


def calculate_fw_derivative(
    sw: float,
    fw: float,
    swi: float,
    sor: float,
    b: float,
) -> float:
    """
    Calculates the partial derivative of fractional flow with respect to water saturation:
    f'w = dfw / dSw.

    Scan References:
    - Scan Page 41, Appendix A, Eq. 4 (and derivation in Appendix B, Scan Page 47):
      f'w = -fw * (1 - fw) * (-b) * [(1 - Sor - Swi) / ((Sw - Swi) * (1 - Sw - Sor))]
    """
    denom = (sw - swi) * (1.0 - sw - sor)
    if denom <= 1e-12 or fw <= 0.0 or fw >= 1.0:
        return 0.0
    num = 1.0 - sor - swi
    return float(fw * (1.0 - fw) * b * (num / denom))


def solve_welge_breakthrough(
    swi: float,
    sor: float,
    a: float,
    b: float,
    mu_o: float,
    mu_w: float,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Executes the Welge tangent search to find breakthrough saturation Swbt, fwbt, f'wbt,
    and average saturation behind the front Sw_avg_bt.

    Scan References:
    - Scan Page 3 ("Calculation of f'w"):
      1. Calculate Sw1 at intersection of fw curve and straight line connecting Swi to 1-Sor (Eq. 7).
      2. Calculate Sw2 at middle of chord connecting Sw1 to 1-Sor (Eq. 8).
      3. Compare slope of fw curve at Sw2 (Slope2, Eq. 9) with secant slope from Swi to Sw2 (Slope3, Eq. 10).
      4. Adjust Sw2 until Slope2 == Slope3, then compute Sw_avg_bt via Eq. 6.
    - Scan Page 27 (Figure 1B): Flowchart for Swbt
    - Scan Page 30 (Figure 2): Graphical illustration of Sw1, Sw2, Slope2, Slope3
    - Scan Pages 41–42 (Appendix A, Eqs. 5–10)
    """
    # Step 1: Eq. 7 (Scan Page 41) - Initial intersection point Sw1
    # At Sw1, the straight diagonal line between Swi and (1 - Sor) intersects fw:
    c_const = ((mu_o / mu_w) / a) ** (1.0 / b)
    sw1 = (swi + c_const * (1.0 - sor)) / (1.0 + c_const)

    # Step 2: Eq. 8 (Scan Page 42) - Midpoint Sw2 between Sw1 and (1 - Sor)
    sw2_init = (sw1 + (1.0 - sor)) / 2.0

    # Step 3: Iterative tangent solver matching Slope2 (Eq. 9) and Slope3 (Eq. 10)
    # Note: Setting Slope2 == Slope3 algebraically yields Eq. 5 (Scan Page 41):
    # Swbt = 1 - Sor - (1 - fwbt) * b * (1 - Sor - Swi)
    low = swi + 1e-5
    high = (1.0 - sor) - 1e-5
    sw_curr = sw2_init
    iterations = 0

    for i in range(200):
        iterations = i + 1
        sw_curr = 0.5 * (low + high)
        fw_curr = calculate_fw_from_sw(sw_curr, swi, sor, a, b, mu_o, mu_w)
        # Eq. 9 (Scan Page 42): Slope2 = f'w at Sw2
        slope2 = calculate_fw_derivative(sw_curr, fw_curr, swi, sor, b)
        # Eq. 10 (Scan Page 42): Slope3 = fw(Sw2) / (Sw2 - Swi)
        slope3 = fw_curr / (sw_curr - swi)

        diff = slope2 - slope3
        if abs(diff) < tol:
            break
        if diff > 0:
            # Curve derivative is steeper than secant line -> move right
            low = sw_curr
        else:
            # Curve derivative is flatter than secant line -> move left
            high = sw_curr

    swbt = float(sw_curr)
    fwbt = calculate_fw_from_sw(swbt, swi, sor, a, b, mu_o, mu_w)
    fw_prime_bt = calculate_fw_derivative(swbt, fwbt, swi, sor, b)
    slope3_final = fwbt / (swbt - swi)

    # Eq. 6 (Scan Page 41): Average water saturation behind front at breakthrough
    # Sw_avg_bt - Swbt = (1 - fwbt) / f'wbt
    sw_avg_bt = swbt + (1.0 - fwbt) / fw_prime_bt

    return {
        "Sw1_initial": float(sw1),
        "Sw2_initial": float(sw2_init),
        "Swbt": swbt,
        "fwbt": float(fwbt),
        "fw_prime_bt": float(fw_prime_bt),
        "Slope2": float(fw_prime_bt),
        "Slope3": float(slope3_final),
        "Sw_avg_bt": float(sw_avg_bt),
        "Qbt_pv": float(1.0 / fw_prime_bt) if fw_prime_bt > 0 else 0.0,
        "iterations": iterations,
    }


def solve_post_breakthrough_state(
    fw: float,
    swi: float,
    sor: float,
    a: float,
    b: float,
    mu_o: float,
    mu_w: float,
) -> Dict[str, float]:
    """
    Calculates water saturation at the producing end (Sw), derivative (f'wabt),
    and average water saturation in the swept zone (Sw_avg) AFTER breakthrough (fw >= fwbt).

    Scan References:
    - Scan Page 3: "After breakthrough, f'wabt and Sw_avg are directly calculated using Eqs. 4 and 6."
    - Scan Page 41 (Appendix A, Eqs. 2, 3, 4, 6)
    """
    if fw >= 1.0 - 1e-9:
        return {"Sw": 1.0 - sor, "Swn": np.inf, "fw_prime": 0.0, "Sw_avg": 1.0 - sor}

    # Invert Eq. 3 to get kro/krw at the producing fractional flow fw:
    kro_krw = ((1.0 - fw) / fw) * (mu_o / mu_w)
    # Invert Eq. 2 (kro/krw = a * Swn^(-b)) to get Swn:
    swn = (kro_krw / a) ** (-1.0 / b)
    # Invert Swn = (Sw - Swi) / (1 - Sw - Sor) to get Sw:
    sw = (swi + swn * (1.0 - sor)) / (1.0 + swn)

    # Eq. 4 (Scan Page 41): f'w at Sw
    fw_prime = calculate_fw_derivative(sw, fw, swi, sor, b)
    # Eq. 6 (Scan Page 41): Sw_avg = Sw + (1 - fw) / f'w
    sw_avg = sw + (1.0 - fw) / fw_prime if fw_prime > 0 else (1.0 - sor)
    sw_avg = min(sw_avg, 1.0 - sor)

    return {
        "Sw": float(sw),
        "Swn": float(swn),
        "fw_prime": float(fw_prime),
        "Sw_avg": float(sw_avg),
    }


def generate_fractional_flow_curve(
    swi: float,
    sor: float,
    a: float,
    b: float,
    mu_o: float,
    mu_w: float,
    swbt: float,
    fwbt: float,
    sw_avg_bt: float,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Generates the 2D curve data for Figure 7 (Scan Page 35, fig7.jpg) and Figure 10 (Scan Page 38),
    including the S-shaped fractional flow curve fw(Sw), its derivative f'w(Sw),
    and the Welge tangent line from (Swi, 0) through (Swbt, fwbt) to (Sw_avg_bt, 1.0).
    """
    sw_grid = np.linspace(swi, 1.0 - sor, n_points)
    tangent_slope = fwbt / (swbt - swi) if (swbt > swi) else 0.0

    rows = []
    for sw in sw_grid:
        fw = calculate_fw_from_sw(float(sw), swi, sor, a, b, mu_o, mu_w)
        fw_p = calculate_fw_derivative(float(sw), fw, swi, sor, b)
        swn = calculate_normalized_saturation(float(sw), swi, sor)
        tangent_val = tangent_slope * (float(sw) - swi) if float(sw) <= sw_avg_bt else np.nan
        rows.append({
            "Sw": float(sw),
            "Swn": float(swn),
            "fw": float(fw),
            "fw_prime": float(fw_p),
            "Welge_Tangent": float(tangent_val) if not np.isnan(tangent_val) else np.nan,
        })

    return pd.DataFrame(rows)
