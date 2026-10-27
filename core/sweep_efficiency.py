# core/sweep_efficiency.py
import numpy as np
from scipy.optimize import brentq
from typing import Dict
from typing import Dict, Any, cast

class SweepEfficiencyEngine:
    """
    Module 3: Mobility Ratio, Areal Sweep Efficiency (Ea) & Vertical Coverage (C).
    Implements Eqs. 11-21 (pp. 42-44) and Table 1 (p. 13) of ARCO TSR 84-28.
    """

    # Table 1 (p. 13): Coefficients in Areal Sweep Efficiency Correlations
    AREAL_COEFFS: Dict[str, Dict[str, float]] = {
        "5-spot": {
            "a1": -0.2062, "a2": -0.0712, "a3": -0.5110,
            "a4":  0.3048, "a5":  0.1230, "a6":  0.4394
        },
        "direct_line": {
            "a1": -0.3014, "a2": -0.1568, "a3": -0.9402,
            "a4":  0.3714, "a5": -0.0865, "a6":  0.8805
        },
        "staggered_line": {
            "a1": -0.2077, "a2": -0.1059, "a3": -0.3526,
            "a4":  0.2608, "a5":  0.2444, "a6":  0.3158
        }
    }

    # Eq. 16 (p. 43): Vertical Coverage Correlation Constants
    COV_A1 = 3.334088568
    COV_A2 = 0.7737348199
    COV_A3 = -1.225859406

    @staticmethod
    def calculate_mobility_ratio(
        krw_sw_bar: float,
        kro_soi: float,
        mu_o: float,
        mu_w: float,
        fw_sw_bar: float = 1.0,
        after_breakthrough: bool = False
    ) -> float:
        """
        Calculates displacing-to-displaced Mobility Ratio M (Eqs. 11-12, pp. 42-43).
        - Before breakthrough (Eq. 11): M = (krw(Sw_bar) * mu_o) / (kro(Soi) * mu_w)
        - After breakthrough (Eq. 12):  M = (1 / fw(Sw_bar)) * (krw(Sw_bar) * mu_o) / (kro(Soi) * mu_w)
        """
        m_base = (krw_sw_bar * mu_o) / (kro_soi * mu_w)
        if after_breakthrough and fw_sw_bar > 0:
            return float(m_base / fw_sw_bar)
        return float(m_base)

    @classmethod
    def calculate_areal_sweep(
        cls,
        m: float,
        fw: float,
        pattern: str = "5-spot"
    ) -> float:
        """
        Calculates Areal Sweep Efficiency Ea (Eqs. 14-15, p. 43).
        Valid for 0 < M <= 10 before and after breakthrough.
        """
        if pattern not in cls.AREAL_COEFFS:
            raise ValueError(f"Pattern must be one of {list(cls.AREAL_COEFFS.keys())}")

        c = cls.AREAL_COEFFS[pattern]
        # Safeguard log arguments for extreme low mobility ratios (M -> 0)
        arg1 = max(m + c["a2"], 1e-6)
        arg2 = max(m + c["a5"], 1e-6)

        y_areal = (c["a1"] * np.log(arg1) + c["a3"]) * fw + c["a4"] * np.log(arg2) + c["a6"]
        if y_areal <= 0:
            return 1.0
        ea = 1.0 / (1.0 + y_areal)
        return float(np.clip(ea, 0.0, 1.0))

    @staticmethod
    def calculate_y_parameter(wor: float, m: float, v: float) -> float:
        """
        Calculates deSouza & Brigham Y correlation parameter (Eq. 17, p. 43).
        """
        f_v = -0.6891 + 0.9735 * v + 1.6453 * (v ** 2)
        numerator = (wor + 0.4) * (18.948 - 2.499 * v)
        denominator = (m + 1.137 - 0.8094 * v) * (10.0 ** f_v)
        return float(numerator / denominator)

    @classmethod
    def calculate_coverage(
        cls,
        wor: float,
        m: float,
        v: float,
        method: str = "HP41_exact",
        tol: float = 1e-5
    ) -> Dict[str, float]:
        """
        Calculates Vertical Coverage C from Y (Eqs. 16-21, pp. 43-44).
        - 'HP41_exact': Uses the 4-regime piecewise rules (Eqs. 18-21, p. 44)
        - 'continuous': Solves Y = a1 * C^a2 * (1 - C)^a3 directly via Brent's method
        """
        y = cls.calculate_y_parameter(wor, m, v)
        a1, a2, a3 = cls.COV_A1, cls.COV_A2, cls.COV_A3

        if method == "HP41_exact":
            if y < 0.5:
                # Regime 1 (Eq. 18)
                c = 0.0
            elif y <= 2.6:
                # Regime 2 (Eq. 19): Fixed-point iteration initialized at C0 = 0.3
                c = 0.3
                for _ in range(200):
                    c_next = (y / (a1 * ((1.0 - c) ** a3))) ** (1.0 / a2)
                    if abs(c_next - c) < tol:
                        c = c_next
                        break
                    c = c_next
            elif y <= 5.0:
                # Regime 3 (Eq. 20): Direct logarithmic fit
                c = (np.log(y) + 0.60414) / 4.2569
            else:
                # Regime 4 (Eq. 21): Fixed-point iteration initialized at C0 = 0.65
                c = 0.65
                for _ in range(200):
                    c_next = 1.0 - ((y / (a1 * (c ** a2))) ** (1.0 / a3))
                    if abs(c_next - c) < tol:
                        c = c_next
                        break
                    c = c_next
        elif method == "continuous":
            def obj(c_val: float) -> float:
                return a1 * (c_val ** a2) * ((1.0 - c_val) ** a3) - y
            root_c = brentq(obj, 1e-9, 1.0 - 1e-9)
            c = float(cast(float, root_c))
        else:
            raise ValueError("method must be 'HP41_exact' or 'continuous'")

        return {
            "Y": round(float(y), 4),
            "C": round(float(np.clip(c, 0.0, 1.0)), 4)
        }