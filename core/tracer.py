"""
core/tracer.py
Radioactive Tracer Injection Design Module (Example 2).

Scan References:
- Scan Pages 6-8: "Example 2. Design of Radioactive Tracer Injection"
  * W_tag = q_i * t_bt * B_w + Sw_avg_bt * V_p * E_a * C
  * Q(pCi) = W_tag * D * m / zeta
  * zeta = exp(-lambda * t_bt), fraction of tracer decay
  * lambda = ln(2) / t_1/2 (1/day)
  * C(pCi/bbl) = Q(pCi) / (q_i * t_bt)
  * %MPC = (C / MPC) * 100
- Scan Page 23 (Table 10): Isotope properties (HTO, Co-60, Co-57, Ni-63)
- Scan Page 25 (Table 12): Tracer concentration data output table
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def calculate_tracer_design(
    tracer_rows: List[Dict[str, Any]],
    qi_bd: float,
    t_bt_days: float,
    bw: float,
    sw_avg_bt: float,
    vp_rb: float,
    ea_bt: float,
    c_bt: float,
    override_w_tag_bbl: Optional[float] = None,
    override_inj_vol_bbl: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes the required radioactive tracer material Q(pCi), injection concentration
    C(pCi/bbl), decay fraction zeta, and percentage of Maximum Permissible Concentration (%MPC)
    matching Scan Page 6 and Table 12 (Scan Page 25).
    """
    # Scan Page 6 equation: W_tag = q_i * t_bt * B_w + Sw_avg_bt * V_p * E_a * C
    inj_vol_bbl = override_inj_vol_bbl if override_inj_vol_bbl is not None else (qi_bd * t_bt_days)
    swept_water_rb = sw_avg_bt * vp_rb * ea_bt * c_bt
    w_tag_calc = inj_vol_bbl * bw + swept_water_rb
    w_tag = override_w_tag_bbl if override_w_tag_bbl is not None else w_tag_calc

    records = []
    for row in tracer_rows:
        name = str(row["Tracer"])
        t_half = float(row["t_half_days"])
        d_limit = float(row["D_pCi_bbl"])
        m_mult = float(row["m"])
        mpc = float(row["MPC_pCi_bbl"])

        # Decay constant lambda = ln(2) / t_1/2 (Scan Page 6)
        lam = np.log(2.0) / t_half if t_half > 0 else 0.0
        # Fraction of tracer remaining at breakthrough: zeta = exp(-lambda * t_bt)
        zeta = float(np.exp(-lam * t_bt_days))

        # Required tracer quantity Q(pCi) = W_tag * D * m / zeta (Scan Page 6)
        q_pci = (w_tag * d_limit * m_mult) / zeta if zeta > 0 else 0.0
        q_ci = q_pci * 1e-12

        # Tracer concentration C(pCi/bbl) and %MPC (Scan Page 7 & Table 12 on Scan Page 25)
        c_pci_bbl = q_pci / inj_vol_bbl if inj_vol_bbl > 0 else 0.0
        pct_mpc = (c_pci_bbl / mpc) * 100.0 if mpc > 0 else 0.0

        records.append({
            "Tracer": name,
            "t_half_days": t_half,
            "lambda_inv_day": float(lam),
            "D_pCi_bbl": d_limit,
            "m": m_mult,
            "zeta": zeta,
            "Q_pCi": float(q_pci),
            "Q_Ci": float(q_ci),
            "C_pCi_bbl": float(c_pci_bbl),
            "MPC_pCi_bbl": mpc,
            "pct_MPC": float(pct_mpc),
        })

    return {
        "t_bt_days": float(t_bt_days),
        "inj_vol_bbl": float(inj_vol_bbl),
        "swept_water_rb": float(swept_water_rb),
        "W_tag_bbl": float(w_tag),
        "tracer_df": pd.DataFrame(records),
    }

