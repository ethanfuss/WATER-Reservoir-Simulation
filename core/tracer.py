# core/tracer.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

class RadioactiveTracerDesigner:
    """
    Module 5: Radioactive Tracer Injection Design & Regulatory Envelope.
    Implements equations on Pages 6-8 and Tables 10-12 (pp. 23-25) of ARCO TSR 84-28.
    """

    # Table 10 (p. 23): Tracer Characteristics (Sample volume = 300 ml)
    TRACER_DB: Dict[str, Dict[str, Union[float, str]]] = {
        "HTO":   {"D_pCi_bbl": 3.18e5, "MPC_pCi_bbl": 4.77e8, "emission": "beta-", "half_life_yr": 12.3},
        "Co-60": {"D_pCi_bbl": 1510.0, "MPC_pCi_bbl": 7.95e6, "emission": "gamma", "half_life_yr": 5.3},
        "Co-57": {"D_pCi_bbl": 1510.0, "MPC_pCi_bbl": 7.95e6, "emission": "gamma", "half_life_yr": 0.743},
        "Ni-63": {"D_pCi_bbl": 7950.0, "MPC_pCi_bbl": 4.77e6, "emission": "beta-", "half_life_yr": 100.0},
    }

    @classmethod
    def design_tracer_program(
        cls,
        qi_bbd: float,
        bw: float,
        t_bt_days: float,
        swept_connate_water_rb: float,
        detection_multiple_m: float = 50.0,
        w_inj_override_rb: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculates Wtag, decay fraction (zeta), required tracer quantity Q (pCi),
        produced concentration C (pCi/bbl), %MPC, and 100% MPC fracture residence time.
        """
        w_inj_rb = w_inj_override_rb if w_inj_override_rb is not None else (qi_bbd * t_bt_days * bw)
        w_tag_rb = w_inj_rb + swept_connate_water_rb

        records = []
        for name, props in cls.TRACER_DB.items():
            d_lim = float(props["D_pCi_bbl"])
            mpc = float(props["MPC_pCi_bbl"])
            t_half_days = float(props["half_life_yr"]) * 365.0

            # Decay constant lambda (1/day) and decay fraction zeta (Page 6)
            lam = np.log(2.0) / t_half_days
            zeta = float(np.exp(-lam * t_bt_days))

            # Total tracer quantity Q (pCi) = Wtag * D * m / zeta (Page 6)
            q_pci = (w_tag_rb * d_lim * detection_multiple_m) / zeta

            # Average concentration C (pCi/bbl) = Q / W_inj (Table 12, p. 25)
            c_pci_bbl = q_pci / w_inj_rb

            # Regulatory %MPC = (C / MPC) * 100 (Page 7)
            pct_mpc = (c_pci_bbl / mpc) * 100.0

            # Minimum safe residence time at 100% MPC in fractured rock: t_min = Q / (qi * MPC) (Page 8)
            t_100pct_mpc_days = q_pci / (qi_bbd * mpc)

            records.append({
                "Tracer": name,
                "Emission": props["emission"],
                "Half_Life_yr": props["half_life_yr"],
                "W_tag_rb": round(w_tag_rb, 0),
                "m": detection_multiple_m,
                "zeta_decay": round(zeta, 4),
                "Q_pCi": q_pci,
                "Q_display": f"{q_pci / 1e12:.2f}e12" if name == "HTO" else f"{q_pci / 1e9:.1f}e9",
                "C_pCi_bbl": round(c_pci_bbl, 1),
                "pct_MPC": round(pct_mpc, 2),
                "Min_Safe_t_bt_days_100MPC": round(t_100pct_mpc_days, 1)
            })

        return pd.DataFrame(records)

    @classmethod
    def evaluate_late_breakthrough_sensitivity(
        cls,
        base_design_df: pd.DataFrame,
        qi_bbd: float,
        bw: float,
        new_t_bt_days: float,
        swept_connate_water_rb: float,
        base_w_tag_rb: float
    ) -> pd.DataFrame:
        """
        Replicates the sensitivity study on Page 8: determines new Detection Limit Multiple m(new),
        concentration C(pCi/bbl), and %MPC if breakthrough takes twice as long (e.g., 644 days).
        """
        w_inj_new_rb = qi_bbd * new_t_bt_days * bw
        w_tag_new_rb = w_inj_new_rb + swept_connate_water_rb

        sens_rows = []
        for _, row in base_design_df.iterrows():
            name = row["Tracer"]
            mpc = float(cls.TRACER_DB[name]["MPC_pCi_bbl"])
            q_pci = float(row["Q_pCi"])
            m_new = float(row["m"]) * (base_w_tag_rb / w_tag_new_rb)
            c_new = q_pci / w_inj_new_rb
            pct_mpc_new = (c_new / mpc) * 100.0

            sens_rows.append({
                "Tracer": name,
                "t_bt_days": new_t_bt_days,
                "W_tag_new_MMrb": round(w_tag_new_rb / 1e6, 3),
                "m_new": round(m_new, 1),
                "C_new_pCi_bbl": round(c_new, 1),
                "pct_MPC_new": round(pct_mpc_new, 2)
            })

        return pd.DataFrame(sens_rows)