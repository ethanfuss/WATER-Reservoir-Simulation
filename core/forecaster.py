"""
core/forecaster.py
Main WATER Simulation Loop: Computes Original Oil in Place (OOIP), Fill-up Time (t_fil),
and iterates across the producing Water-Oil Ratio (WOR) schedule to calculate Oil Recovery (Np),
Cumulative Water Produced (Wp via Trapezoidal Rule), Water Injected (Wi), Time (t), and Oil Rate (Qo).

Scan References:
- Scan Page 5 ("Calculation of Oil Recovery"): Trapezoidal rule integration over WOR steps
- Scan Page 15 (Table 3, Step 2): OOIP and time for fill-up displayed
- Scan Page 20 (Table 7, table7.jpg): Oil Recovery Calculations for Example 1
- Scan Page 36 (Figure 8): Printer output of WATER
- Scan Page 37 (Figure 9, fig9.jpg): WOR vs. Oil Recovery (% OOIP)
- Scan Pages 44–45 (Appendix A, Eqs. 22–27): Np, Fwo, Wp, Wi, t, Qo
- Scan Page 52 (Appendix C, Lines 53–75): Exact RPN formulas for OOIP and TFIL
"""

from typing import List, Dict, Any
import pandas as pd
from .frac_flow import calculate_fw_from_sw, solve_post_breakthrough_state
from .sweep_efficiency import (
    calculate_mobility_ratio,
    calculate_areal_sweep,
    solve_vertical_coverage,
)


def run_waterflood_forecast(
    vp_rb: float,
    swi: float,
    sor: float,
    sgi: float,
    kro_swi: float,
    mu_o: float,
    mu_w: float,
    bo: float,
    bw: float,
    iw_bd: float,
    pattern: str,
    v: float,
    a: float,
    b: float,
    swbt: float,
    fwbt: float,
    sw_avg_bt: float,
    wor_schedule: List[float],
    rel_perm_rows: List[Dict[str, Any]],
    after_bt_m_method: str = "endpoint_krw",
) -> Dict[str, Any]:
    """
    Executes the full multi-step waterflood performance forecast across all target WOR values.
    """
    # 1. Initial Reservoir Volumetrics (Appendix C, Lines 53–75, Scan Page 52)
    soi = 1.0 - swi - sgi
    ooip_stb = (vp_rb * soi) / bo
    # Fill-up volume and time to fill up gas space (Scan Page 52, Lines 64–75)
    v_fillup_rb = sgi * vp_rb
    t_fillup_years = v_fillup_rb / (iw_bd * bw * 365.0) if (iw_bd * bw) > 0 else 0.0

    # 2. Iterate through increasing values of WOR (Scan Page 5 & Table 7 on Scan Page 20)
    schedule_records = []
    wp_cum_stb = 0.0
    prev_fwo = 0.0
    prev_np_stb = 0.0
    prev_t_years = t_fillup_years

    for idx, wor in enumerate(sorted(wor_schedule)):
        # Fractional flow corresponding to reservoir WOR: fw = WOR / (1 + WOR) (Table 7, Col 2)
        fw_wor = wor / (1.0 + wor)

        # Layering of logic Before vs. After Breakthrough (Scan Page 3, 20, 41–43):
        # When fw_wor < fwbt, the flood front in the swept layers is at breakthrough conditions
        # (Sw_avg = Sw_avg_bt). When fw_wor >= fwbt, the swept layers are after breakthrough
        # and Sw_avg increases with fw_wor via Eqs. 4 & 6.
        is_after_bt = bool(fw_wor >= fwbt)

        if not is_after_bt:
            sw_front = swbt
            sw_avg = sw_avg_bt
            fw_for_m = fwbt
            phase_label = "Pre-BT Front (fw < fwbt)"
        else:
            post_state = solve_post_breakthrough_state(fw_wor, swi, sor, a, b, mu_o, mu_w)
            sw_front = post_state["Sw"]
            sw_avg = post_state["Sw_avg"]
            fw_for_m = calculate_fw_from_sw(sw_avg, swi, sor, a, b, mu_o, mu_w)
            phase_label = "Post-BT (fw >= fwbt)"

        # Mobile oil saturation change in swept zone (Table 7 Col 4: dSo = Sw_avg - Swi - Sgi)
        delta_so = max(sw_avg - swi - sgi, 0.0)

        # Mobility Ratio M (Eqs. 11–13, Scan Pages 42–43)
        m_res = calculate_mobility_ratio(
            sw_avg=sw_avg,
            fw_at_sw_avg=fw_for_m,
            is_after_breakthrough=is_after_bt,
            rel_perm_rows=rel_perm_rows,
            swi=swi,
            sor=sor,
            kro_swi=kro_swi,
            mu_o=mu_o,
            mu_w=mu_w,
            after_bt_method=after_bt_m_method,
        )
        m_val = m_res["M"]

        # Areal Sweep Efficiency Ea (Eqs. 14–15, Scan Page 43)
        ea_res = calculate_areal_sweep(m=m_val, fw=fw_wor, pattern=pattern)
        ea_val = ea_res["Ea"]

        # Vertical Coverage C (Eqs. 16–21, Scan Pages 43–44)
        c_res = solve_vertical_coverage(wor=wor, m=m_val, v=v)
        c_val = c_res["C"]
        y_cov = c_res["Y"]

        # Volumetric recovery factor product (Table 7 Col 8: dSo * Ea * C)
        dso_ea_c = delta_so * ea_val * c_val

        # Eq. 22 (Scan Page 44): Cumulative Oil Produced Np (STB)
        np_stb = (delta_so / bo) * ea_val * c_val * vp_rb
        ooip_pct = (np_stb / ooip_stb) * 100.0 if ooip_stb > 0 else 0.0

        # Eq. 23 (Scan Page 44) & Table 7 Col 10: Surface Water-Oil Ratio Fwo (or Fwp)
        fwo_surface = wor * (bo / bw)

        # Eq. 24 (Scan Page 45): Cumulative Water Produced Wp via Trapezoidal Rule
        delta_np = max(np_stb - prev_np_stb, 0.0)
        if idx == 0:
            wp_step = 0.5 * (fwo_surface + 0.0) * delta_np
        else:
            wp_step = 0.5 * (fwo_surface + prev_fwo) * delta_np
        wp_cum_stb += wp_step

        # Eq. 25 (Scan Page 45): Cumulative Water Injected Wi (RB)
        # Note: Includes initial gas fill-up volume v_fillup_rb plus displaced oil & produced water
        wi_rb = v_fillup_rb + bw * wp_cum_stb + bo * np_stb

        # Eq. 26 (Scan Page 45): Project Time t (years)
        t_years = wi_rb / (iw_bd * 365.0) if iw_bd > 0 else 0.0
        t_days = t_years * 365.0

        # Eq. 27 (Scan Page 45): Incremental Oil Production Rate Qo (STB/D)
        delta_t_years = t_years - prev_t_years
        qo_stbd = delta_np / (365.0 * delta_t_years) if delta_t_years > 1e-9 else 0.0

        schedule_records.append({
            "WOR": float(wor),
            "fw": float(fw_wor),
            "Phase": phase_label,
            "Sw_front": float(sw_front),
            "Sw": float(sw_avg),           # Matches 'Sw' column in Table 7 (table7.jpg)
            "dSo": float(delta_so),        # Matches 'ΔSo' column in Table 7
            "M": float(m_val),             # Matches 'M' column in Table 7
            "Ea": float(ea_val),           # Matches 'Ea' column in Table 7
            "Y_cov": float(y_cov),
            "C": float(c_val),             # Matches 'C' column in Table 7
            "Coverage_Regime": c_res["regime"],
            "dSo_Ea_C": float(dso_ea_c),   # Matches 'ΔSo·Ea·C' column in Table 7
            "OOIP_pct": float(ooip_pct),   # Matches '%OOIP' column in Table 7
            "Fwp": float(fwo_surface),     # Matches 'Fwp' (Eq. 23 Fwo) column in Table 7
            "Np_STB": float(np_stb),       # Eq. 22
            "Wp_STB": float(wp_cum_stb),   # Eq. 24
            "Wi_RB": float(wi_rb),         # Eq. 25
            "Time_years": float(t_years),  # Eq. 26
            "Time_days": float(t_days),
            "Qo_STBD": float(qo_stbd),     # Eq. 27
        })

        prev_fwo = fwo_surface
        prev_np_stb = np_stb
        prev_t_years = t_years

    df_schedule = pd.DataFrame(schedule_records)

    # Breakthrough state summary for Tracer Design (Scan Page 6)
    m_bt_res = calculate_mobility_ratio(
        sw_avg=sw_avg_bt,
        fw_at_sw_avg=fwbt,
        is_after_breakthrough=False,
        rel_perm_rows=rel_perm_rows,
        swi=swi,
        sor=sor,
        kro_swi=kro_swi,
        mu_o=mu_o,
        mu_w=mu_w,
    )
    ea_bt = calculate_areal_sweep(m=m_bt_res["M"], fw=fwbt, pattern=pattern)["Ea"]
    wor_bt = fwbt / (1.0 - fwbt) if fwbt < 1.0 else 100.0
    c_bt = solve_vertical_coverage(wor=wor_bt, m=m_bt_res["M"], v=v)["C"]
    np_bt_stb = (max(sw_avg_bt - swi - sgi, 0.0) / bo) * ea_bt * c_bt * vp_rb
    wi_bt_rb = v_fillup_rb + bo * np_bt_stb
    t_bt_days = wi_bt_rb / iw_bd if iw_bd > 0 else 0.0

    return {
        "OOIP_STB": float(ooip_stb),
        "SOI": float(soi),
        "V_fillup_RB": float(v_fillup_rb),
        "t_fillup_years": float(t_fillup_years),
        "t_fillup_days": float(t_fillup_years * 365.0),
        "M_bt": float(m_bt_res["M"]),
        "Ea_bt": float(ea_bt),
        "C_bt": float(c_bt),
        "Np_bt_STB": float(np_bt_stb),
        "Wi_bt_RB": float(wi_bt_rb),
        "t_bt_days": float(t_bt_days),
        "t_bt_years": float(t_bt_days / 365.0),
        "schedule_df": df_schedule,
    }
