# core/rock_fluid.py
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from typing import List, Dict, Optional, Union, Any


class RockFluidProperties:
    """
    Module 1: Reservoir Initialization, Heterogeneity & Rock-Fluid Characterization.
    Replicates HP-41 main initialization (Appendix C, p. 52; Table 3, p. 15) and
    subroutines KVA, SWN, and LIN1 from ARCO TSR 84-28.
    """

    @staticmethod
    def convert_acft_to_rb(ac_ft: float, porosity: float = 1.0) -> float:
        """
        Converts Acre-Feet to Reservoir Barrels (1 Ac-ft = 7,758 bbl).
        In Example 2 (p. 6), Vp = 220 Ac-ft = 1,706,760 rb.
        """
        return float(ac_ft * 7758.0 * porosity)

    @staticmethod
    def initialize_reservoir(
        vp_rb: float,
        swi: float,
        sor: float,
        sgi: float,
        bo: float,
        bw: float,
        iw_bbd: float
    ) -> Dict[str, float]:
        """
        Replicates lines 09-14 and 50-75 of Appendix C (p. 52) and Table 3 Step 2 (p. 15).
        Computes movable saturation range (Reg 20), OOIP (Reg 42), and Fill-up Time (Reg 43).
        """
        movable_sat_range = 1.0 - swi - sor
        ooip_stb = (vp_rb / bo) * (1.0 - swi - sgi)
        t_fillup_yr = (sgi * vp_rb) / (iw_bbd * bw * 365.0) if (iw_bbd * bw) > 0 else 0.0

        return {
            "movable_sat_range_reg20": round(float(movable_sat_range), 4),
            "OOIP_STB": round(float(ooip_stb), 2),
            "t_fillup_years": round(float(t_fillup_yr), 4)
        }

    @staticmethod
    def calculate_permeability_variation(
        permeabilities: Optional[Union[List[float], np.ndarray]] = None,
        manual_v: Optional[float] = None,
        ddof: int = 1
    ) -> Dict[str, float]:
        """
        Computes Dykstra-Parsons Permeability Variation (V) via Subroutine KVA (Eq. 1, p. 41)
        or accepts direct manual V entry per Flowchart 1A (p. 26).
        """
        if manual_v is not None:
            return {
                "V": round(float(manual_v), 4),
                "k50": np.nan,
                "k84_1": np.nan,
                "mean_log_k": np.nan,
                "std_log_k": np.nan
            }

        if permeabilities is None or len(permeabilities) < 2:
            raise ValueError("Provide at least 2 permeability values or specify manual_v.")

        k = np.asarray(permeabilities, dtype=float)
        if np.any(k <= 0):
            raise ValueError("Permeability values must be strictly positive (> 0 md).")

        log_k = np.log10(k)
        mean_log_k = float(np.mean(log_k))
        std_log_k = float(np.std(log_k, ddof=ddof))

        k_50 = 10.0 ** mean_log_k
        k_841 = 10.0 ** (mean_log_k - std_log_k)
        v = 1.0 - (10.0 ** (-std_log_k))

        return {
            "V": round(float(v), 4),
            "k50": round(float(k_50), 3),
            "k84_1": round(float(k_841), 3),
            "mean_log_k": round(mean_log_k, 5),
            "std_log_k": round(std_log_k, 5)
        }

    @staticmethod
    def calculate_swn(
        sw: Union[float, np.ndarray],
        swi: float,
        sor: float,
        formulation: str = "TSR"
    ) -> np.ndarray:
        """
        Calculates normalized water saturation (Swn) per Subroutine SWN.
        - 'TSR': (Sw - Swi) / (1 - Sw - Sor) [Required by TSR 84-28 Eq. 2 & Eq. 4]
        - 'standard': (Sw - Swi) / (1 - Swi - Sor)
        """
        sw_arr = np.asarray(sw, dtype=float)
        if formulation == "TSR":
            denom = 1.0 - sw_arr - sor
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(denom > 0, (sw_arr - swi) / denom, np.nan)
        elif formulation == "standard":
            return (sw_arr - swi) / (1.0 - swi - sor)
        raise ValueError("formulation must be 'TSR' or 'standard'.")

    @classmethod
    def fit_relative_permeability(
        cls,
        sw: Optional[Union[List[float], np.ndarray]] = None,
        kro: Optional[Union[List[float], np.ndarray]] = None,
        krw: Optional[Union[List[float], np.ndarray]] = None,
        swi: float = 0.0,
        sor: float = 0.0,
        manual_a: Optional[float] = None,
        manual_b: Optional[float] = None,
        formulation: str = "TSR",
        points_slice: Optional[slice] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Fits kro/krw = a * Swn^(-b) via log-linear regression (Subroutine LIN1)
        or accepts manual (a, b) per Flowchart 1A (p. 26).
        """
        if manual_a is not None and manual_b is not None:
            return {
                "a": round(float(manual_a), 4),
                "b": round(float(manual_b), 4),
                "r_squared": 1.0,
                "swn_fitted": np.array([]),
                "ratio_fitted": np.array([])
            }

        sw_arr = np.asarray(sw, dtype=float)
        kro_arr = np.asarray(kro, dtype=float)
        krw_arr = np.asarray(krw, dtype=float)

        if points_slice is not None:
            sw_arr = sw_arr[points_slice]
            kro_arr = kro_arr[points_slice]
            krw_arr = krw_arr[points_slice]

        swn = cls.calculate_swn(sw_arr, swi, sor, formulation=formulation)
        valid = (swn > 0) & (kro_arr > 0) & (krw_arr > 0) & (~np.isnan(swn))

        if np.sum(valid) < 2:
            raise ValueError("At least 2 valid saturation points are required for regression.")

        swn_v = swn[valid]
        ratio_v = kro_arr[valid] / krw_arr[valid]

        res: Any = linregress(np.log10(swn_v), np.log10(ratio_v))
        slope = float(res.slope)
        intercept = float(res.intercept)
        r_val = float(res.rvalue)

        a = 10.0 ** intercept
        b = -slope
        r_squared = r_val ** 2

        return {
            "a": round(a, 4),
            "b": round(b, 4),
            "r_squared": round(r_squared, 5),
            "swn_fitted": np.round(swn_v, 4),
            "ratio_fitted": np.round(ratio_v, 4)
        }

    @staticmethod
    def build_krw_interpolator(
        sw: Union[List[float], np.ndarray],
        krw: Union[List[float], np.ndarray],
        swi: float,
        sor: float
    ):
        """
        Constructs an automated interpolator for krw(Sw_bar) across WOR steps.
        """
        sw_arr = np.asarray(sw, dtype=float)
        krw_arr = np.asarray(krw, dtype=float)
        return interp1d(
            sw_arr,
            krw_arr,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate" , # type: ignore[arg-type]
        )  