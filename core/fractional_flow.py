# core/fractional_flow.py
import numpy as np
from scipy.optimize import brentq
from typing import Dict, Optional, Union
from typing import Dict, Any, cast

class FractionalFlowEngine:
    """
    Module 2: Buckley-Leverett Fractional Flow & Welge Tangent Solver.
    Implements Eqs. 3-10 (pp. 41-42) and Post-Breakthrough Saturation State (p. 3, Eq. 23).
    """

    def __init__(
        self,
        a: float,
        b: float,
        swi: float,
        sor: float,
        mu_o: float,
        mu_w: float,
        bo: float = 1.0,
        bw: float = 1.0
    ):
        self.a = float(a)
        self.b = float(b)
        self.swi = float(swi)
        self.sor = float(sor)
        self.mu_o = float(mu_o)
        self.mu_w = float(mu_w)
        self.bo = float(bo)
        self.bw = float(bw)
        self.max_sw = 1.0 - self.sor

    def swn(self, sw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Normalized saturation per TSR Eq. 2: (Sw - Swi) / (1 - Sw - Sor)."""
        return (sw - self.swi) / (1.0 - sw - self.sor)

    def kro_krw_ratio(self, sw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Relative permeability ratio kro/krw = a * Swn^(-b) (Eq. 2, p. 41)."""
        swn_val = self.swn(sw)
        return self.a * np.power(swn_val, -self.b)

    def fractional_flow(self, sw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Fractional flow of water fw (Eq. 3, p. 41)."""
        sw_arr = np.asarray(sw, dtype=float)
        out = np.zeros_like(sw_arr, dtype=float)
        mask_mid = (sw_arr > self.swi) & (sw_arr < self.max_sw)
        mask_high = sw_arr >= self.max_sw
        if np.any(mask_mid):
            ratio = self.kro_krw_ratio(sw_arr[mask_mid])
            out[mask_mid] = 1.0 / (1.0 + ratio * (self.mu_w / self.mu_o))
        out[mask_high] = 1.0
        return float(out) if np.ndim(sw) == 0 else out

    def fractional_flow_derivative(self, sw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Analytical derivative dfw/dSw (Eq. 4, p. 41; Appendix B, p. 47)."""
        fw = self.fractional_flow(sw)
        term = (1.0 - self.sor - self.swi) / ((sw - self.swi) * (1.0 - sw - self.sor))
        return fw * (1.0 - fw) * self.b * term

    def sw_from_fw(self, fw: float) -> float:
        """
        Analytically inverts Eq. 2 and Eq. 3 to find front saturation Sw from fw (Eq. 6, p. 41).
        Because kro/krw = (1/fw - 1)*(mu_o/mu_w) = a * Swn^(-b), Swn is solved in closed form.
        """
        if fw <= 0.0:
            return self.swi
        if fw >= 1.0:
            return self.max_sw
        ratio = ((1.0 - fw) / fw) * (self.mu_o / self.mu_w)
        swn_val = (ratio / self.a) ** (-1.0 / self.b)
        return (self.swi + swn_val * (1.0 - self.sor)) / (1.0 + swn_val)

    def solve_breakthrough(
        self,
        method: str = "continuous",
        manual_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Solves Welge tangent breakthrough parameters (Swbt, fwbt, dfw_dSw_bt, Sw_bar_bt).
        Supports:
          - manual_override: Direct entry per Table 3 Step 5 (p. 16)
          - 'continuous': Exact root-finding via scipy.optimize.brentq
          - 'HP41_exact': Chord-bisection loop per Page 3 and Flowchart 1B (p. 27)
        """
        if manual_override is not None:
            return manual_override

        # Eq. 7 (p. 41): Intersection of fw curve with diagonal line connecting Swi to 1-Sor
        c_const = ((self.a * self.mu_w) / self.mu_o) ** (1.0 / (self.b - 1.0))
        sw1 = (c_const * (1.0 - self.sor) + self.swi) / (1.0 + c_const)

        if method == "HP41_exact":
            # Eq. 8 (p. 42): Midpoint of chord connecting Sw1 to 1-Sor
            sw2 = (sw1 + 1.0 - self.sor) / 2.0
            for _ in range(500):
                slope2 = self.fractional_flow_derivative(sw2)  # Eq. 9
                slope3 = self.fractional_flow(sw2) / (sw2 - self.swi)  # Eq. 10
                diff = slope3 - slope2
                if abs(diff) < 0.0005:
                    break
                # Adjust Sw2 based on slope difference (Flowchart 1B, p. 27)
                sw2 += 0.05 * diff * (self.max_sw - self.swi)
                sw2 = float(np.clip(sw2, self.swi + 1e-5, self.max_sw - 1e-5))
            swbt = sw2
        else:
            def tangent_residual(sw_val: float) -> float:
                val = self.fractional_flow_derivative(sw_val) - (
                    self.fractional_flow(sw_val) / (sw_val - self.swi)
                )
                return float(np.squeeze(val))
            # Search above inflection region up to 1 - Sor
            sw_grid = np.linspace(self.swi + 1e-4, self.max_sw - 1e-4, 200)
            derivs = self.fractional_flow_derivative(sw_grid)
            peak_sw = float(sw_grid[np.argmax(derivs)])
            root_val = brentq(tangent_residual, peak_sw, self.max_sw - 1e-6)
            swbt = float(cast(float, root_val))

        fwbt = float(self.fractional_flow(swbt))
        fwp_bt = float(self.fractional_flow_derivative(swbt))
        # Eq. 5 (p. 41): Sw_bar - Swbt = (1 - fwbt) / f'wbt
        sw_bar_bt = swbt + (1.0 - fwbt) / fwp_bt
        # Reservoir Fwo = fw / (1 - fw); Surface WOR = Fwo * (Bw / Bo) (Eq. 23, p. 44)
        fwo_bt = fwbt / (1.0 - fwbt)
        wor_bt = fwo_bt * (self.bw / self.bo)

        return {
            "Sw1_initial": round(float(sw1), 4),
            "Swbt": round(swbt, 4),
            "fwbt": round(fwbt, 4),
            "fwp_bt": round(fwp_bt, 4),
            "Sw_bar_bt": round(sw_bar_bt, 4),
            "WOR_bt": round(wor_bt, 3)
        }

    def state_at_wor(self, wor: float, bt_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Determines saturation state at any given surface producing WOR (Table 3 Step 6, p. 16).
        - Before/at breakthrough (fw <= fwbt): front is at Swbt, average saturation is Sw_bar_bt,
          and flowing fw used in areal sweep is fw = Fwo / (1 + Fwo).
        - After breakthrough (fw > fwbt): calculates Sw2 from Eq. 6, f'w from Eq. 4, and Sw_bar from Eq. 5.
        """
        if bt_params is None:
            bt_params = self.solve_breakthrough()

        # Eq. 23 (p. 44): Fwo = WOR * (Bo / Bw)
        fwo = wor * (self.bo / self.bw)
        fw_wor = fwo / (1.0 + fwo) if fwo > 0 else 0.0

        if fw_wor <= bt_params["fwbt"]:
            return {
                "is_after_bt": False,
                "WOR": float(wor),
                "Fwo": round(float(fwo), 4),
                "fw": round(float(fw_wor), 4),
                "Sw_front": bt_params["Swbt"],
                "fwp": bt_params["fwp_bt"],
                "Sw_bar": bt_params["Sw_bar_bt"]
            }
        else:
            sw_front = self.sw_from_fw(fw_wor)
            fwp = float(self.fractional_flow_derivative(sw_front))
            sw_bar = sw_front + (1.0 - fw_wor) / fwp
            return {
                "is_after_bt": True,
                "WOR": float(wor),
                "Fwo": round(float(fwo), 4),
                "fw": round(float(fw_wor), 4),
                "Sw_front": round(float(sw_front), 4),
                "fwp": round(float(fwp), 4),
                "Sw_bar": round(float(sw_bar), 4)
            }