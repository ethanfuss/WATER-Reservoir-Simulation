"""
core/coverage.py
Calculates the de Souza-Brigham parameter Y and solves for Vertical Coverage (C)
using Fassihi's closed-form correlation (TSR 84-28 Scan Pages 43-44, Eqs. 16-21).
"""

import math
from typing import Dict
from core.presets import COVERAGE_CONSTANTS


def calculate_coverage_parameter_y(wor: float, m: float, v: float) -> float:
    """
    Computes parameter Y using Eq. 17 (Scan Page 43).
    Denominator exponent evaluated as: +1.6453*V^2 + 0.9735*V - 0.6891
    """
    numerator = (wor + 0.4) * (18.948 - 2.499 * v)
    den_m_term = m + 1.137 - 0.8094 * v
    den_exp = -0.6891 + 0.9735 * v + 1.6453 * (v**2)
    denominator = den_m_term * (10.0**den_exp)

    return float(numerator / denominator)


def solve_vertical_coverage(
    y: float,
    constants: Dict[str, float] = COVERAGE_CONSTANTS,
    tol: float = 1.0e-4,
    max_iter: int = 100,
) -> float:
    """
    Solves for Vertical Coverage C from Y = a1 * C^a2 * (1 - C)^a3.
    Follows piecewise iteration guidelines on Scan Page 44 (Eqs. 18-21):
      - If Y < 0.5: C = 0.0
      - If 0.5 <= Y <= 2.6: C_{i+1} = [ Y / (a1 * (1 - C_i)^a3) ]^(1/a2), starting at C0 = 0.30
      - If 2.6 < Y <= 5.0: C = (ln(Y) + 0.60414) / 4.2569
      - If Y > 5.0: C_{i+1} = 1.0 - [ Y / (a1 * C_i^a2) ]^(1/a3), starting at C0 = 0.65
    """
    a1 = constants["a1"]
    a2 = constants["a2"]
    a3 = constants["a3"]

    if y < 0.5:
        return 0.0

    if 2.6 < y <= 5.0:
        c_direct = (math.log(y) + 0.60414) / 4.2569
        return float(max(0.0, min(1.0, c_direct)))

    if 0.5 <= y <= 2.6:
        c_curr = 0.30
        for _ in range(max_iter):
            # Guard against boundary saturation
            c_safe = max(1.0e-5, min(0.9999, c_curr))
            inner = y / (a1 * ((1.0 - c_safe) ** a3))
            if inner <= 0.0:
                break
            c_next = inner ** (1.0 / a2)
            if abs(c_next - c_curr) <= tol:
                return float(max(0.0, min(1.0, c_next)))
            c_curr = 0.5 * (c_curr + c_next)  # Damped step
        return float(max(0.0, min(1.0, c_curr)))

    # Y > 5.0
    c_curr = 0.65
    for _ in range(max_iter):
        c_safe = max(1.0e-5, min(0.9999, c_curr))
        inner = y / (a1 * (c_safe**a2))
        if inner <= 0.0:
            break
        c_next = 1.0 - (inner ** (1.0 / a3))
        if abs(c_next - c_curr) <= tol:
            return float(max(0.0, min(1.0, c_next)))
        c_curr = 0.5 * (c_curr + c_next)  # Damped step

    return float(max(0.0, min(1.0, c_curr)))