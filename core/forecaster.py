# core/forecaster.py
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from core.rock_fluid import RockFluidProperties
from core.fractional_flow import FractionalFlowEngine
from core.sweep_efficiency import SweepEfficiencyEngine

class WaterfloodForecaster:
    """
    Module 4: Performance Forecaster & Material Balance Integrator.
    Implements Eqs. 22-27 (pp. 44-45) and replicates Table 7 (p. 20) & Table 11 (p. 24).
    """

    def __init__(
        self,
        vp_rb: float,
        swi: float,
        sor: float,
        sgi: float,
        bo: float,
        bw: float,
        mu_o: float,
        mu_w: float,
        kro_soi: float,
        iw_bbd: float,
        v: float,
        a: float,
        b: float,
        pattern: str = "5-spot",
        krw_bt: float = 0.1,
        krw_interpolator: Optional[Callable[[float], float]] = None
    ):
        self.vp_rb = float(vp_rb)
        self.swi = float(swi)
        self.sor = float(sor)
        self.sgi = float(sgi)
        self.bo = float(bo)
        self.bw = float(bw)
        self.mu_o = float(mu_o)
        self.mu_w = float(mu_w)
        self.kro_soi = float(kro_soi)
        self.iw_bbd = float(iw_bbd)
        self.v = float(v)
        self.pattern = pattern
        self.krw_bt = float(krw_bt)
        self.krw_interpolator = krw_interpolator

        self.init_metrics = RockFluidProperties.initialize_reservoir(
            vp_rb, swi, sor, sgi, bo, bw, iw_bbd
        )
        self.frac_engine = FractionalFlowEngine(a, b, swi, sor, mu_o, mu_w, bo, bw)

    def run_forecast(
        self,
        wor_steps: List[float],
        krw_manual_map: Optional[Dict[float, float]] = None,
        solver_mode: str = "HP41_exact",
        after_bt_mobility_mode: str = "total"
    ) -> pd.DataFrame:
        """
        Executes step-by-step waterflood forecast across target WOR steps.
        - after_bt_mobility_mode:
            'total': Uses Eq. 12 on p. 43 (divides by fw(Sw_bar), as in Example 2 Table 11)
            'water_only': Uses Eq. 11 at Sw_bar without dividing by fw (as in Example 1)
        """
        bt = self.frac_engine.solve_breakthrough(
            method="HP41_exact" if solver_mode == "HP41_exact" else "continuous"
        )
        ooip = self.init_metrics["OOIP_STB"]
        w_fillup_rb = self.sgi * self.vp_rb

        rows = []
        wp_cum_stb = 0.0
        prev_np_stb = 0.0
        prev_fwo = 0.0
        prev_t_yr = self.init_metrics["t_fillup_years"]

        for idx, wor in enumerate(wor_steps):
            state = self.frac_engine.state_at_wor(wor, bt_params=bt)
            sw_bar = state["Sw_bar"]
            is_after_bt = state["is_after_bt"]

            # Determine krw at Sw_bar (Table 3 Step 7, p. 16)
            if krw_manual_map and wor in krw_manual_map:
                krw_val = float(krw_manual_map[wor])
            elif not is_after_bt:
                krw_val = self.krw_bt
            elif self.krw_interpolator is not None:
                krw_val = float(self.krw_interpolator(sw_bar))
            else:
                krw_val = self.krw_bt

            # Mobility Ratio M (Eqs. 11-12, pp. 42-43)
            fw_at_sw_bar = float(self.frac_engine.fractional_flow(sw_bar))
            use_div_fw = (after_bt_mobility_mode == "total")
            m = SweepEfficiencyEngine.calculate_mobility_ratio(
                krw_sw_bar=krw_val,
                kro_soi=self.kro_soi,
                mu_o=self.mu_o,
                mu_w=self.mu_w,
                fw_sw_bar=fw_at_sw_bar if use_div_fw else 1.0,
                after_breakthrough=True if use_div_fw else False
            )

            # Areal Sweep Ea (Eqs. 14-15, p. 43) & Vertical Coverage C (Eqs. 16-21, pp. 43-44)
            ea = SweepEfficiencyEngine.calculate_areal_sweep(m, state["fw"], self.pattern)
            cov = SweepEfficiencyEngine.calculate_coverage(wor, m, self.v, method=solver_mode)
            c = cov["C"]

            # Cumulative Oil Produced Np (Eq. 22, p. 44)
            delta_sw = max(0.0, sw_bar - self.swi - self.sgi)
            np_stb = (delta_sw / self.bo) * ea * c * self.vp_rb

            # Enforce physical monotonicity if first step is very low WOR
            if idx > 0 and np_stb < prev_np_stb:
                np_stb = prev_np_stb

            # Reservoir Water-Oil Ratio Fwo (Eq. 23, p. 44) & Cumulative Water Wp (Eq. 24, p. 45)
            fwo = wor * (self.bo / self.bw)
            if idx == 0:
                wp_cum_stb = 0.5 * fwo * np_stb
            else:
                wp_cum_stb += 0.5 * (fwo + prev_fwo) * (np_stb - prev_np_stb)

            # Cumulative Water Injected Wi (Eq. 25, p. 45) & Time t (Eq. 26, p. 45)
            wi_rb = w_fillup_rb + wp_cum_stb * self.bw + self.bo * np_stb
            t_yr = wi_rb / (self.iw_bbd * self.bw * 365.0)

            # Oil Production Rate Qo (Eq. 27, p. 45)
            dt_yr = t_yr - prev_t_yr
            qo_stbd = ((np_stb - prev_np_stb) / (365.0 * dt_yr)) if dt_yr > 1e-6 else 0.0

            # Swept water volume (Sw_bar * C * Ea * Vp) for Tracer Module (Table 11, p. 24)
            swept_water_rb = sw_bar * c * ea * self.vp_rb

            rows.append({
                "WOR": wor,
                "Stage": "After BT" if is_after_bt else "Pre/At BT",
                "Sw_front": state["Sw_front"],
                "Sw_bar": round(sw_bar, 4),
                "fw": state["fw"],
                "krw_Sw_bar": round(krw_val, 4),
                "M": round(m, 4),
                "Ea": round(ea, 4),
                "Y": cov["Y"],
                "C": round(c, 4),
                "Np_STB": round(np_stb, 1),
                "Recovery_pct_OOIP": round((np_stb / ooip) * 100.0, 2) if ooip > 0 else 0.0,
                "Wp_STB": round(wp_cum_stb, 1),
                "Wi_rb": round(wi_rb, 1),
                "Time_yr": round(t_yr, 3),
                "Qo_STBD": round(qo_stbd, 2),
                "Swept_Water_MMrb": round(swept_water_rb / 1e6, 4)
            })

            prev_np_stb = np_stb
            prev_fwo = fwo
            prev_t_yr = t_yr

        return pd.DataFrame(rows)