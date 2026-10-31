"""
app.py
Streamlit Web Application for the WATER 3-D Single-Grid Waterflooding Simulator.
Follows Technical Service Report TSR 84-28 (Scan Pages 1–56) and SPE 13945 (Pages 604–606).
"""

from typing import Any, Dict, List, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    PATTERN_COEFFICIENTS,
    COVERAGE_CONSTANTS,
    get_example_1_preset,
    get_example_2_preset,
    TABLE_2_BENCHMARK,
    EXAMPLE_1_TABLE_7_MANUAL,
    EXAMPLE_1_CRAIG_CURVE,
    calculate_permeability_variation,
    calculate_normalized_saturation,
    fit_rel_perm_power_law,
    build_rel_perm_table,
    solve_welge_breakthrough,
    generate_fractional_flow_curve,
    calculate_areal_sweep,
    solve_vertical_coverage,
    run_waterflood_forecast,
    calculate_tracer_design,
)

st.set_page_config(
    page_title="WATER Simulator (TSR 84-28)",
    page_icon="🛢️",
    layout="wide",
)

st.title("🛢️ WATER: 3-D Single-Grid Waterflooding Simulator")
st.caption(
    "Exact implementation of Technical Service Report **TSR 84-28** (*M. R. Fassihi & W. J. O'Brien*) "
    "and **SPE 13945**. All scan page, table, equation, and figure references map directly to the original report scans."
)

# ---------------------------------------------------------------------------
# Sidebar: Preset Selection & Interactive Reservoir Inputs (Table 3, Scan Page 15)
# ---------------------------------------------------------------------------
st.sidebar.header("1. Preset & Pattern Selection")
preset_choice = st.sidebar.radio(
    "Select Reservoir Configuration (Scan Pages 5–6):",
    options=[
        "Example 1 (Five-Spot General Waterflood)",
        "Example 2 (Staggered Line Drive & Tracer Design)",
        "Custom Reservoir",
    ],
    index=0,
)

if preset_choice.startswith("Example 1"):
    active_preset = get_example_1_preset()
elif preset_choice.startswith("Example 2"):
    active_preset = get_example_2_preset()
else:
    active_preset = get_example_1_preset()

pattern_options = list(PATTERN_COEFFICIENTS.keys())
default_pat_idx = (
    pattern_options.index(active_preset["pattern"])
    if active_preset["pattern"] in pattern_options
    else 0
)
pattern = st.sidebar.selectbox(
    "Flooding Pattern (Scan Page 13, Table 1; Fig. 1A):",
    options=pattern_options,
    index=default_pat_idx,
)

st.sidebar.header("2. Reservoir & Fluid Properties")
st.sidebar.caption("Matches Table 3 (Scan Page 15) & e1input.jpg (Scan Page 5)")

swi = st.sidebar.slider(
    "Initial Water Saturation, Swi (PV)",
    min_value=0.01,
    max_value=0.60,
    value=float(active_preset["swi"]),
    step=0.01,
)
sor = st.sidebar.slider(
    "Residual Oil Saturation, Sor (PV)",
    min_value=0.05,
    max_value=0.60,
    value=float(active_preset["sor"]),
    step=0.01,
)
sgi = st.sidebar.slider(
    "Initial Gas Saturation, Sgi (PV)",
    min_value=0.00,
    max_value=0.40,
    value=float(active_preset["sgi"]),
    step=0.01,
)

kro_swi = st.sidebar.number_input(
    "Rel. Perm of Oil at Swi, kro(Soi)",
    min_value=0.01,
    max_value=1.00,
    value=float(active_preset["kro_swi"]),
    step=0.01,
    format="%.3f",
)
k_nominal = st.sidebar.number_input(
    "Nominal Reservoir Permeability, k (md)",
    min_value=0.1,
    max_value=10000.0,
    value=float(active_preset["k_nominal_md"]),
    step=1.0,
)
vp_rb = st.sidebar.number_input(
    "Reservoir Pore Volume, Vp (RB)",
    min_value=100.0,
    max_value=1.0e9,
    value=float(active_preset["vp_rb"]),
    step=1000.0,
    format="%.1f",
)
iw_bd = st.sidebar.number_input(
    "Water Injection Rate, iw (B/D)",
    min_value=1.0,
    max_value=1.0e6,
    value=float(active_preset["iw_bd"]),
    step=10.0,
    format="%.1f",
)

col_mu1, col_mu2 = st.sidebar.columns(2)
with col_mu1:
    mu_o = st.number_input(
        "Oil Visc. μo (cp)",
        min_value=0.05,
        max_value=1000.0,
        value=float(active_preset["mu_o_cp"]),
        step=0.1,
    )
    bo = st.number_input(
        "Oil FVF, Bo",
        min_value=0.80,
        max_value=3.00,
        value=float(active_preset["bo"]),
        step=0.05,
    )
with col_mu2:
    mu_w = st.number_input(
        "Water Visc. μw (cp)",
        min_value=0.05,
        max_value=100.0,
        value=float(active_preset["mu_w_cp"]),
        step=0.05,
    )
    bw = st.number_input(
        "Water FVF, Bw",
        min_value=0.80,
        max_value=2.00,
        value=float(active_preset["bw"]),
        step=0.01,
    )

# ---------------------------------------------------------------------------
# Sidebar Section 3: Technical Branching Options (Flowchart Fig. 1A, Scan Page 26)
# ---------------------------------------------------------------------------
st.sidebar.header("3. Flowchart Calculation Modes")
st.sidebar.caption("Follows Scan Page 3 & Flowchart Figure 1A (Scan Page 26)")

v_mode = st.sidebar.radio(
    "Permeability Variation (V) Mode (Scan Page 3):",
    options=[
        "Estimate from Core Layers (Subroutine KVA)",
        "Enter V Directly",
    ],
    index=0,
)
if v_mode == "Enter V Directly":
    v_direct_input = st.sidebar.slider(
        "Direct Permeability Variation, V (0.3 < V < 0.8)",
        min_value=0.05,
        max_value=0.95,
        value=float(active_preset["v_direct"]),
        step=0.01,
    )
else:
    v_direct_input = float(active_preset["v_direct"])

ab_mode = st.sidebar.radio(
    "Rel-Perm Correlation (a, b) Mode (Scan Page 3 & 26):",
    options=[
        "Estimate from Rel-Perm Table (Suboutines SWN & LIN1)",
        "Enter a and b Directly",
    ],
    index=0,
)
if ab_mode == "Enter a and b Directly":
    a_direct_input = st.sidebar.number_input(
        "Direct Coefficient a (Eq. 2)",
        min_value=0.01,
        max_value=100.0,
        value=float(active_preset["a_direct"]),
        format="%.4f",
    )
    b_direct_input = st.sidebar.number_input(
        "Direct Exponent b (Eq. 2)",
        min_value=0.10,
        max_value=10.0,
        value=float(active_preset["b_direct"]),
        format="%.4f",
    )
else:
    a_direct_input = float(active_preset["a_direct"])
    b_direct_input = float(active_preset["b_direct"])

swbt_mode = st.sidebar.radio(
    "Breakthrough Saturation (Swbt) Mode (Scan Page 26, Fig. 1A):",
    options=[
        "Compute via Welge Tangent (Subroutine SWBT)",
        "Enter Swbt, Sw_avg_bt, fwbt Directly",
    ],
    index=0,
)
if swbt_mode != "Compute via Welge Tangent (Subroutine SWBT)":
    swbt_direct_in = st.sidebar.number_input(
        "Direct Swbt",
        min_value=0.10,
        max_value=0.90,
        value=float(active_preset["swbt_direct"]),
        format="%.3f",
    )
    sw_avg_bt_direct_in = st.sidebar.number_input(
        "Direct Sw_avg_bt",
        min_value=0.10,
        max_value=0.95,
        value=float(active_preset["sw_avg_bt_direct"]),
        format="%.3f",
    )
    fwbt_direct_in = st.sidebar.number_input(
        "Direct fwbt",
        min_value=0.05,
        max_value=0.99,
        value=float(active_preset["fwbt_direct"]),
        format="%.3f",
    )
else:
    swbt_direct_in = float(active_preset["swbt_direct"])
    sw_avg_bt_direct_in = float(active_preset["sw_avg_bt_direct"])
    fwbt_direct_in = float(active_preset["fwbt_direct"])

after_bt_m_method = st.sidebar.selectbox(
    "Post-Breakthrough Mobility Ratio Method (Scan Page 20 vs Page 43):",
    options=[
        "endpoint_krw (Table 7 Method: krw(Sw_avg)*μo / (kro(Soi)*μw))",
        "eq12_total_mobility (Eq. 12: Includes 1/fw(Sw_avg) multiplier)",
    ],
    index=0,
)
m_method_key = (
    "endpoint_krw"
    if after_bt_m_method.startswith("endpoint_krw")
    else "eq12_total_mobility"
)

wor_str = st.sidebar.text_input(
    "WOR Schedule (comma-separated, Scan Page 4 & Table 7):",
    value=", ".join(str(x) for x in active_preset["wor_schedule"]),
)
try:
    wor_schedule = [float(x.strip()) for x in wor_str.split(",") if x.strip()]
except ValueError:
    wor_schedule = active_preset["wor_schedule"]

# ---------------------------------------------------------------------------
# Editable Input Tables (Table 5 & Table 6 for Ex 1; Table 8 & Table 9 for Ex 2)
# ---------------------------------------------------------------------------
with st.expander(
    "📋 View / Edit Input Tables: Relative Permeability (Table 5 / 8) & Core Permeability Layers (Table 6 / 9)",
    expanded=False,
):
    col_t5, col_t6 = st.columns(2)
    with col_t5:
        st.markdown(
            "**Relative Permeability Input Data (Scan Page 18, `table5.jpg` / Scan Page 21, Table 8)**"
        )
        st.caption(
            "Check `fit_include` to include a row in the power-law regression "
            "$\\frac{k_{ro}}{k_{rw}} = a S_{wn}^{-b}$ (matches the pencil bracket in `table5.jpg`)."
        )
        edited_rel_perm_df = st.data_editor(
            pd.DataFrame(active_preset["rel_perm_table"]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"rel_perm_editor_{preset_choice}",
        )
    with col_t6:
        st.markdown(
            "**Core Layer Permeabilities (Scan Page 19, `table6.jpg` / Scan Page 22, Table 9)**"
        )
        st.caption(
            "Used by Subroutine KVA (Eq. 1, Scan Page 41) to estimate permeability variation $V$."
        )
        edited_core_df = st.data_editor(
            pd.DataFrame(active_preset["core_table"]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"core_editor_{preset_choice}",
        )

rel_perm_rows = cast(List[Dict[str, Any]], edited_rel_perm_df.to_dict("records"))
core_rows = cast(List[Dict[str, Any]], edited_core_df.to_dict("records"))

# ---------------------------------------------------------------------------
# Execute Sequential Mathematical Pipeline (Modules 1 -> 5)
# ---------------------------------------------------------------------------
# 1. Permeability Variation V (Scan Page 3 & 41, Eq. 1)
kva_result = calculate_permeability_variation(core_rows)
v_val = kva_result["V"] if v_mode.startswith("Estimate") else v_direct_input

# 2. Relative Permeability Power-Law Fit a, b (Scan Page 3 & 41, Eq. 2)
lin1_result = fit_rel_perm_power_law(rel_perm_rows, swi=swi, sor=sor)
if ab_mode.startswith("Estimate"):
    a_val = lin1_result["a"]
    b_val = lin1_result["b"]
else:
    a_val = a_direct_input
    b_val = b_direct_input

rel_perm_calc_df = build_rel_perm_table(
    rel_perm_rows=rel_perm_rows,
    swi=swi,
    sor=sor,
    mu_o=mu_o,
    mu_w=mu_w,
    a=a_val,
    b=b_val,
)

# 3. Welge Tangent & Breakthrough Saturation (Scan Page 3, 35, 41–42, Eqs. 4–10)
welge_result = solve_welge_breakthrough(
    swi=swi,
    sor=sor,
    a=a_val,
    b=b_val,
    mu_o=mu_o,
    mu_w=mu_w,
)
if swbt_mode.startswith("Compute"):
    swbt_val = welge_result["Swbt"]
    fwbt_val = welge_result["fwbt"]
    sw_avg_bt_val = welge_result["Sw_avg_bt"]
    fw_prime_bt_val = welge_result["fw_prime_bt"]
else:
    swbt_val = swbt_direct_in
    fwbt_val = fwbt_direct_in
    sw_avg_bt_val = sw_avg_bt_direct_in
    fw_prime_bt_val = (
        (1.0 - fwbt_val) / (sw_avg_bt_val - swbt_val)
        if (sw_avg_bt_val > swbt_val)
        else welge_result["fw_prime_bt"]
    )

fw_curve_df = generate_fractional_flow_curve(
    swi=swi,
    sor=sor,
    a=a_val,
    b=b_val,
    mu_o=mu_o,
    mu_w=mu_w,
    swbt=swbt_val,
    fwbt=fwbt_val,
    sw_avg_bt=sw_avg_bt_val,
)

# 4. Full Waterflood Recovery Schedule (Scan Page 5, 20, 44–45, Eqs. 22–27)
forecast_result = run_waterflood_forecast(
    vp_rb=vp_rb,
    swi=swi,
    sor=sor,
    sgi=sgi,
    kro_swi=kro_swi,
    mu_o=mu_o,
    mu_w=mu_w,
    bo=bo,
    bw=bw,
    iw_bd=iw_bd,
    pattern=pattern,
    v=v_val,
    a=a_val,
    b=b_val,
    swbt=swbt_val,
    fwbt=fwbt_val,
    sw_avg_bt=sw_avg_bt_val,
    wor_schedule=wor_schedule,
    rel_perm_rows=rel_perm_rows,
    after_bt_m_method=m_method_key,
)
schedule_df = forecast_result["schedule_df"]

# ---------------------------------------------------------------------------
# Main Dashboard Navigation Tabs (Including Dedicated calcs.jpg Tab)
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🧮 Key Calculations (calcs.jpg)",
    "📈 Fractional Flow & Rel-Perm (Figs. 6, 7, 10 & Table 5)",
    "🪨 Permeability & Sweep Efficiencies (Figs. 3, 4, 5 & Tables 1, 2, 6)",
    "🛢️ Oil Recovery & Table 7 / Fig. 9 Comparison",
    "☢️ Radioactive Tracer Design (Example 2, Table 12)",
])

# ===========================================================================
# TAB 1: Dedicated Key Calculations Tab (Matching calcs.jpg, Scan Page 2)
# ===========================================================================
with tabs[0]:
    st.subheader("Key Program Calculations (calcs.jpg — Scan Page 2)")
    st.markdown(
        "As outlined on **Scan Page 2 (`calcs.jpg`)**, the **WATER** simulator is a single-grid, 3-D "
        "waterflooding simulator built around five core engineering calculations. Below is the live "
        "step-by-step mathematical ledger for your current inputs."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "1. Fractional Flow (fwbt)",
        f"{fwbt_val:.3f}",
        f"Swbt = {swbt_val:.3f}, Sw_avg = {sw_avg_bt_val:.3f}",
    )
    m2.metric(
        "2. Areal Sweep at BT (Eabt)",
        f"{forecast_result['Ea_bt']:.3f}",
        f"Pattern: {pattern}",
    )
    m3.metric(
        "3. Perm. Variation (V)",
        f"{v_val:.3f}",
        f"k50 = {kva_result['k50']:.1f} md",
    )
    m4.metric(
        "4. Coverage at BT (Cbt)",
        f"{forecast_result['C_bt']:.3f}",
        f"M_bt = {forecast_result['M_bt']:.2f}",
    )
    m5.metric(
        "5. OOIP & Fill-Up Time",
        f"{forecast_result['OOIP_STB']:,.0f} STB",
        f"t_fil = {forecast_result['t_fillup_years']:.3f} yrs",
    )

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### 1. Fractional Flow & Welge Breakthrough (Scan Pages 3, 41–42)")
        st.latex(
            r"S_{wn} = \frac{S_w - S_{wi}}{1 - S_w - S_{or}}, \quad "
            r"\frac{k_{ro}}{k_{rw}} = a \, S_{wn}^{-b}, \quad "
            r"f_w = \frac{1}{1 + \left(\frac{k_{ro}}{k_{rw}}\right)\left(\frac{\mu_w}{\mu_o}\right)}"
        )
        st.latex(
            r"f'_w = f_w (1 - f_w) b \left[\frac{1 - S_{or} - S_{wi}}{(S_w - S_{wi})(1 - S_w - S_{or})}\right], \quad "
            r"\overline{S}_w = S_w + \frac{1 - f_w}{f'_w}"
        )
        st.write(
            f"- **Fitted Rel-Perm Constants (Eq. 2):** $a = {a_val:.4f}$, $b = {b_val:.4f}$ ($R^2 = {lin1_result['r_squared']:.4f}$)\n"
            f"- **Welge Initial Estimates (Eqs. 7–8):** $S_{{w1}} = {welge_result['Sw1_initial']:.4f}$, $S_{{w2}} = {welge_result['Sw2_initial']:.4f}$\n"
            f"- **Breakthrough Front (Eqs. 5–6):** $S_{{wbt}} = {swbt_val:.3f}$, $f_{{wbt}} = {fwbt_val:.3f}$, "
            f"$f'_{{wbt}} = {fw_prime_bt_val:.3f}$, $\\overline{{S}}_{{wbt}} = {sw_avg_bt_val:.3f}$"
        )

        st.markdown("#### 2. Areal Sweep Efficiency, $E_a$ (Scan Pages 4, 13, 43)")
        st.latex(r"\frac{1 - E_a}{E_a} = \left[a_1 \ln(M + a_2) + a_3\right] f_w + a_4 \ln(M + a_5) + a_6")
        pat_c = PATTERN_COEFFICIENTS[pattern]
        st.write(
            f"- **Pattern (`Table 1`, Scan Page 13):** `{pattern}` ($a_1={pat_c['a1']}, a_2={pat_c['a2']}, a_3={pat_c['a3']}, "
            f"a_4={pat_c['a4']}, a_5={pat_c['a5']}, a_6={pat_c['a6']}$)\n"
            f"- **Before Breakthrough ($f_w < f_{{wbt}}$):** Uses constant $\\overline{{S}}_{{wbt}} = {sw_avg_bt_val:.3f} \\implies M = {forecast_result['M_bt']:.3f}$\n"
            f"- **After Breakthrough ($f_w \\ge f_{{wbt}}$):** $\\overline{{S}}_w$ and $M$ increase dynamically with $WOR$ (Eqs. 11–13)"
        )

    with c_right:
        st.markdown("#### 3. Permeability Variation, $V$ (Scan Pages 2–3, 41)")
        st.latex(
            r"V = \frac{k_{50} - k_{84.1}}{k_{50}} = \frac{10^{\overline{\log k}} - 10^{\overline{\log k} - \sigma}}{10^{\overline{\log k}}}"
        )
        st.write(
            f"- **Mode (`Scan Page 3`):** `{v_mode}`\n"
            f"- **Log-Normal Parameters (Eq. 1):** $\\overline{{\\log k}} = {kva_result['mean_log_k']:.4f}$, "
            f"$\\sigma = {kva_result['sigma_log_k']:.4f}$, $k_{{50}} = {kva_result['k50']:.2f}\\text{{ md}}$, "
            f"$k_{{84.1}} = {kva_result['k84_1']:.2f}\\text{{ md}} \\implies V = {v_val:.3f}$"
        )

        st.markdown("#### 4. Vertical Coverage, $C$ (Scan Pages 4–5, 14, 43–44)")
        st.latex(
            r"Y = \frac{(WOR + 0.4)(18.948 - 2.499V)}{(M + 1.137 - 0.8094V)\cdot 10^{-0.6891 + 0.9735V + 1.6453V^2}}, \quad "
            r"Y = a_1 C^{a_2}(1 - C)^{a_3}"
        )
        st.write(
            f"- **Fassihi Regression Constants (Eq. 16):** $a_1 = {COVERAGE_CONSTANTS['a1']}$, "
            f"$a_2 = {COVERAGE_CONSTANTS['a2']}$, $a_3 = {COVERAGE_CONSTANTS['a3']}$\n"
            f"- **Piecewise Solver (Eqs. 18–21, Scan Page 44):** Automatically switches across $Y < 0.5$, $0.5 \\le Y \\le 2.6$, $2.6 < Y \\le 5$, and $Y > 5$."
        )

        st.markdown("#### 5. Oil Recovery & Material Balance (Scan Pages 5, 44–45, 52)")
        st.latex(
            r"\text{OOIP} = \frac{V_p(1 - S_{wi} - S_{gi})}{B_o}, \quad "
            r"t_{fil} = \frac{S_{gi} V_p}{i_w B_w \cdot 365}, \quad "
            r"N_p = \frac{\Delta S_w}{B_o} E_a C V_p"
        )
        st.write(
            f"- **Initial Oil in Place (Scan Page 52, Line 62):** $\\text{{OOIP}} = {forecast_result['OOIP_STB']:,.1f}\\text{{ STB}}$\n"
            f"- **Time to Fill Up Gas Space (Scan Page 52, Line 74):** $t_{{fil}} = {forecast_result['t_fillup_years']:.4f}\\text{{ years}}$ ({forecast_result['t_fillup_days']:.1f} days)"
        )

# ===========================================================================
# TAB 2: Fractional Flow & Rel-Perm (Figs. 6, 7, 10 & Table 5)
# ===========================================================================
with tabs[1]:
    st.subheader(
        "Intermediate Graphs: Relative Permeability Ratio (Fig. 6) & Fractional Flow (Figs. 7 & 10)"
    )
    col_g6, col_g7 = st.columns(2)

    with col_g6:
        st.markdown(
            "##### Figure 6: Permeability Ratio ($k_{ro}/k_{rw}$) vs. Normalized Saturation ($S_{wn}$) (Scan Page 34)"
        )
        valid_rp = rel_perm_calc_df.dropna(subset=["Swn", "kro_krw_actual"])
        swn_line = np.logspace(-1, 2, 100)
        ratio_line = a_val * (swn_line ** (-b_val))

        fig6 = go.Figure()
        fig6.add_trace(
            go.Scatter(
                x=valid_rp["Swn"],
                y=valid_rp["kro_krw_actual"],
                mode="markers",
                name="Actual Data (Table 5/8)",
                marker=dict(
                    size=10, symbol="circle-open", line=dict(width=2, color="#1f77b4")
                ),
            )
        )
        fig6.add_trace(
            go.Scatter(
                x=valid_rp["Swn"],
                y=valid_rp["kro_krw_corr"],
                mode="markers",
                name="Correlated Points (Eq. 2)",
                marker=dict(
                    size=9, symbol="square-open", line=dict(width=2, color="#ff7f0e")
                ),
            )
        )
        fig6.add_trace(
            go.Scatter(
                x=swn_line,
                y=ratio_line,
                mode="lines",
                name=f"Fit: a={a_val:.3f}, b={b_val:.3f}",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig6.update_layout(
            xaxis=dict(title="NORMALIZED SATURATION, Swn", type="log", range=[-1, 3]),
            yaxis=dict(title="PERMEABILITY RATIO, kro/krw", type="log", range=[-2, 1]),
            height=460,
            template="plotly_white",
            legend=dict(x=0.45, y=0.95),
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col_g7:
        fig_num_label = (
            "Figure 7 (Ex. 1, Scan Page 35)"
            if not preset_choice.startswith("Example 2")
            else "Figure 10 (Ex. 2, Scan Page 38)"
        )
        st.markdown(
            f"##### {fig_num_label}: Fractional Flow of Water ($f_w$) vs. Water Saturation ($S_w$)"
        )
        fig7 = go.Figure()
        fig7.add_trace(
            go.Scatter(
                x=fw_curve_df["Sw"],
                y=fw_curve_df["fw"],
                mode="lines",
                name="Fractional Flow Curve (Eq. 3)",
                line=dict(color="#1f77b4", width=2.5),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=rel_perm_calc_df["Sw"],
                y=rel_perm_calc_df["fw_actual"],
                mode="markers",
                name="Lab Data Points (Table 5/8)",
                marker=dict(
                    size=9, symbol="circle-open", line=dict(width=2, color="#1f77b4")
                ),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=[swi, swbt_val, sw_avg_bt_val],
                y=[0.0, fwbt_val, 1.0],
                mode="lines+markers",
                name="Welge Tangent Line (Eqs. 5–10)",
                line=dict(color="#d62728", width=2, dash="solid"),
                marker=dict(size=8, color="#d62728"),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=[0.0, swbt_val, swbt_val],
                y=[fwbt_val, fwbt_val, 0.0],
                mode="lines",
                name=f"Swbt={swbt_val:.3f}, fwbt={fwbt_val:.3f}",
                line=dict(color="gray", width=1.5, dash="dot"),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=[sw_avg_bt_val, sw_avg_bt_val],
                y=[0.0, 1.0],
                mode="lines",
                name=f"Sw_avg_bt={sw_avg_bt_val:.3f}",
                line=dict(color="green", width=1.5, dash="dot"),
            )
        )
        fig7.update_layout(
            xaxis=dict(title="WATER SATURATION, Sw", range=[0.0, 0.8]),
            yaxis=dict(title="FRACTIONAL FLOW OF WATER, fw", range=[0.0, 1.05]),
            height=460,
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown(
        "##### Calculated Relative Permeability & Fractional Flow Table (Matches `table5.jpg`, Scan Page 18)"
    )
    st.dataframe(
        rel_perm_calc_df.style.format(
            {
                "Sw": "{:.2f}",
                "Swn": "{:.3f}",
                "kro": "{:.3f}",
                "krw": "{:.3f}",
                "kro_krw_actual": "{:.3f}",
                "kro_krw_corr": "{:.3f}",
                "fw_actual": "{:.4f}",
                "fw_corr": "{:.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
    )

# ===========================================================================
# TAB 3: Permeability & Sweep Efficiencies (Figs. 3, 4, 5 & Tables 1, 2, 6)
# ===========================================================================
with tabs[2]:
    st.subheader(
        "Permeability Distribution (Table 6), Areal Sweep (Fig. 3), & Vertical Coverage (Figs. 4–5, Table 2)"
    )
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(
            "##### Log-Normal Core Permeability Distribution (Scan Page 3 & Table 6, Scan Page 19)"
        )
        layer_df = kva_result["layer_df"]
        fig_perm = go.Figure()
        fig_perm.add_trace(
            go.Scatter(
                x=layer_df["Pct_Greater"],
                y=layer_df["k_md"],
                mode="markers+lines",
                name="Core Layer Permeabilities",
                marker=dict(size=9, color="#2ca02c"),
            )
        )
        fig_perm.add_hline(
            y=kva_result["k50"],
            line_dash="dash",
            line_color="blue",
            annotation_text=f"k50 = {kva_result['k50']:.2f} md",
        )
        fig_perm.add_hline(
            y=kva_result["k84_1"],
            line_dash="dot",
            line_color="red",
            annotation_text=f"k84.1 = {kva_result['k84_1']:.2f} md",
        )
        fig_perm.update_layout(
            xaxis=dict(title="CUMULATIVE PERCENT GREATER THAN (%)", range=[0, 100]),
            yaxis=dict(title="PERMEABILITY, k (md)", type="log"),
            height=420,
            template="plotly_white",
        )
        st.plotly_chart(fig_perm, use_container_width=True)
        st.dataframe(
            layer_df[
                ["Layer_Rank", "k_md", "Thickness_ft", "log10_k", "Pct_Greater"]
            ].style.format(
                {
                    "k_md": "{:.2f}",
                    "Thickness_ft": "{:.1f}",
                    "log10_k": "{:.4f}",
                    "Pct_Greater": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

    with col_p2:
        st.markdown(
            "##### Figure 5: Vertical Coverage ($C$) vs. Correlation Parameter ($Y$) (Scan Page 33 & SPE 13945 Fig. 2)"
        )
        c_grid = np.linspace(0.01, 0.995, 200)
        y_grid = (
            COVERAGE_CONSTANTS["a1"]
            * (c_grid ** COVERAGE_CONSTANTS["a2"])
            * ((1.0 - c_grid) ** COVERAGE_CONSTANTS["a3"])
        )
        fig5 = go.Figure()
        fig5.add_trace(
            go.Scatter(
                x=c_grid,
                y=y_grid,
                mode="lines",
                name="Fassihi Correlation (Eq. 16)",
                line=dict(color="#1f77b4", width=2.5),
            )
        )
        fig5.add_trace(
            go.Scatter(
                x=schedule_df["C"],
                y=schedule_df["Y_cov"],
                mode="markers",
                name="Current Reservoir WOR Steps",
                marker=dict(size=10, color="#d62728", symbol="circle"),
            )
        )
        fig5.update_layout(
            xaxis=dict(title="VERTICAL COVERAGE, C", range=[0.0, 1.0]),
            yaxis=dict(title="CORRELATION PARAMETER, Y", type="log", range=[-1, 3.5]),
            height=420,
            template="plotly_white",
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown(
            "##### Figure 3: Areal Sweep Efficiency ($E_a$) vs. Correlation Function $Y_{areal}$ (Scan Page 31)"
        )
        y_areal_grid = np.linspace(0.0, 1.5, 100)
        ea_grid = 100.0 / (1.0 + y_areal_grid)
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=y_areal_grid,
                y=ea_grid,
                mode="lines",
                name="Eq. 14: Ea = 1 / (1 + Y)",
                line=dict(color="#ff7f0e", width=2.5),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=(1.0 - schedule_df["Ea"]) / schedule_df["Ea"],
                y=schedule_df["Ea"] * 100.0,
                mode="markers",
                name="Current Reservoir WOR Steps",
                marker=dict(size=10, color="#2ca02c", symbol="diamond"),
            )
        )
        fig3.update_layout(
            xaxis=dict(title="CORRELATION FUNCTION, (1 - Ea) / Ea", range=[0.0, 1.5]),
            yaxis=dict(title="AREAL SWEEP EFFICIENCY, Ea (%)", range=[40.0, 100.0]),
            height=420,
            template="plotly_white",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown(
            "##### Table 1: Areal Sweep Coefficients (Scan Page 13 & SPE 13945 Table 2)"
        )
        t1_df = (
            pd.DataFrame(PATTERN_COEFFICIENTS)
            .T.reset_index()
            .rename(columns={"index": "Pattern"})
        )
        st.dataframe(
            t1_df.style.format(
                {
                    c: "{:.4f}"
                    for c in ["a1", "a2", "a3", "a4", "a5", "a6"]
                    if c in t1_df.columns
                }
            ),
            use_container_width=True,
        )
    with col_t2:
        st.markdown(
            "##### Table 2: Vertical Coverage Benchmark Comparison (Scan Page 14 & SPE 13945 Table 1)"
        )
        t2_df = pd.DataFrame(TABLE_2_BENCHMARK).copy()
        if {"WOR", "M", "V"}.issubset(t2_df.columns) and "C_calc" not in t2_df.columns:
            t2_df["C_calc"] = [
                solve_vertical_coverage(wor=row["WOR"], m=row["M"], v=row["V"])["C"]
                for _, row in t2_df.iterrows()
            ]
        st.dataframe(t2_df, use_container_width=True, height=280)

# ===========================================================================
# TAB 4: Oil Recovery & Table 7 / Fig. 9 Comparison (Scan Pages 20, 36, 37)
# ===========================================================================
with tabs[3]:
    st.subheader(
        "Oil Recovery Forecast, Printer Tape Output (Fig. 8), & Benchmark Comparison (Table 7 & Fig. 9)"
    )

    col_fig9, col_rate = st.columns(2)

    with col_fig9:
        st.markdown(
            "##### Figure 9: Water-Oil Ratio (WOR) vs. Oil Recovery (% OOIP) (Scan Page 37)"
        )
        fig9 = go.Figure()

        sim_x_col = (
            schedule_df["OOIP_pct"]
            if "OOIP_pct" in schedule_df.columns
            else (
                schedule_df["Recovery_pct_OOIP"]
                if "Recovery_pct_OOIP" in schedule_df.columns
                else (schedule_df["Np_STB"] / forecast_result["OOIP_STB"] * 100.0)
            )
        )

        fig9.add_trace(
            go.Scatter(
                x=sim_x_col,
                y=schedule_df["WOR"],
                mode="lines+markers",
                name="WATER Simulator Results (Fig. 8)",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(
                    size=9, symbol="circle-open", line=dict(width=2, color="#1f77b4")
                ),
            )
        )

        if preset_choice.startswith("Example 1"):
            manual_df = pd.DataFrame(EXAMPLE_1_TABLE_7_MANUAL)
            craig_df = pd.DataFrame(EXAMPLE_1_CRAIG_CURVE)

            man_x_col = next(
                (
                    c
                    for c in [
                        "OOIP_pct",
                        "Recovery_pct_OOIP",
                        "pct_ooip",
                        "recovery_pct",
                        "Oil_Recovery_pct",
                    ]
                    if c in manual_df.columns
                ),
                None,
            )
            man_y_col = next(
                (c for c in ["WOR", "wor"] if c in manual_df.columns), None
            )
            if man_x_col is None and "Np_STB" in manual_df.columns:
                manual_df["OOIP_pct"] = (
                    manual_df["Np_STB"] / forecast_result["OOIP_STB"] * 100.0
                )
                man_x_col = "OOIP_pct"

            if man_x_col and man_y_col:
                fig9.add_trace(
                    go.Scatter(
                        x=manual_df[man_x_col],
                        y=manual_df[man_y_col],
                        mode="markers",
                        name="Manual Calculations (Table 7)",
                        marker=dict(size=10, symbol="triangle-up", color="#2ca02c"),
                    )
                )

            craig_x_col = next(
                (
                    c
                    for c in [
                        "OOIP_pct",
                        "Recovery_pct_OOIP",
                        "pct_ooip",
                        "recovery_pct",
                        "Oil_Recovery_pct",
                    ]
                    if c in craig_df.columns
                ),
                None,
            )
            craig_y_col = next(
                (c for c in ["WOR", "wor"] if c in craig_df.columns), None
            )
            if craig_x_col and craig_y_col:
                fig9.add_trace(
                    go.Scatter(
                        x=craig_df[craig_x_col],
                        y=craig_df[craig_y_col],
                        mode="lines+markers",
                        name="Craig's Monograph Data (5-Layer)",
                        line=dict(color="#d62728", width=2, dash="dash"),
                        marker=dict(
                            size=9,
                            symbol="triangle-up-open",
                            line=dict(width=2, color="#d62728"),
                        ),
                    )
                )

        fig9.update_layout(
            xaxis=dict(title="OIL RECOVERY, % OOIP", range=[0.0, 60.0]),
            yaxis=dict(
                title="WATER-OIL RATIO, WOR",
                range=[0.0, max(40.0, float(schedule_df["WOR"].max()) * 1.05)],
            ),
            height=460,
            template="plotly_white",
            legend=dict(x=0.03, y=0.97),
        )
        st.plotly_chart(fig9, use_container_width=True)

    with col_rate:
        st.markdown(
            "##### Cumulative Oil Recovery ($N_p$) & Oil Rate ($Q_o$) vs. Project Time ($t$) (Eqs. 22–27, Scan Pages 44–45)"
        )
        t_col = next(
            (c for c in ["t_years", "Time_years", "t"] if c in schedule_df.columns),
            schedule_df.columns[-1],
        )
        np_col = next(
            (c for c in ["Np_STB", "Np"] if c in schedule_df.columns), "Np_STB"
        )
        qo_col = next(
            (c for c in ["Qo_STBD", "Qo_bd", "Qo"] if c in schedule_df.columns), None
        )

        fig_prod = go.Figure()
        fig_prod.add_trace(
            go.Scatter(
                x=schedule_df[t_col],
                y=schedule_df[np_col],
                mode="lines+markers",
                name="Cumulative Oil, Np (STB)",
                line=dict(color="#1f77b4", width=2.5),
                yaxis="y1",
            )
        )
        if qo_col:
            fig_prod.add_trace(
                go.Scatter(
                    x=schedule_df[t_col],
                    y=schedule_df[qo_col],
                    mode="lines+markers",
                    name="Oil Rate, Qo (STB/D)",
                    line=dict(color="#ff7f0e", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
        fig_prod.update_layout(
            xaxis=dict(title="PROJECT TIME, t (years)"),
            yaxis=dict(title="CUMULATIVE OIL PRODUCED, Np (STB)", side="left"),
            yaxis2=dict(
                title="OIL PRODUCTION RATE, Qo (STB/D)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            height=460,
            template="plotly_white",
            legend=dict(x=0.03, y=0.97),
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown(
        "##### Simulated Waterflood Forecast Ledger (Matches Table 7, Scan Page 20 & Fig. 8 Printer Output, Scan Page 36)"
    )
    st.dataframe(schedule_df, use_container_width=True)

    if preset_choice.startswith("Example 1"):
        with st.expander(
            "📖 View Original Report Table 7 Hand-Calculation Benchmark (Scan Page 20)",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(EXAMPLE_1_TABLE_7_MANUAL), use_container_width=True
            )

# ===========================================================================
# TAB 5: Radioactive Tracer Design (Example 2, Scan Pages 6–8, 23–25)
# ===========================================================================
with tabs[4]:
    st.subheader(
        "Radioactive Tracer Injection Design (Example 2 — Scan Pages 6–8 & Tables 10–12, Scan Pages 23–25)"
    )
    st.markdown(
        "Designs tracer slugs (**HTO**, **Co-60**, **Co-57**, **Ni-63**) using breakthrough timing and swept connate water volume "
        "to ensure laboratory detectability ($m \\times D^*$) while remaining safely below the regulatory Maximum Permissible Concentration (**%MPC**)."
    )

    col_tr1, col_tr2, col_tr3 = st.columns(3)
    with col_tr1:
        m_multiple = st.number_input(
            "Detection Limit Multiple, m (Scan Page 7 & Table 12)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
        )
    with col_tr2:
        t_bt_override = st.number_input(
            "Breakthrough Time, t_bt (days) (Scan Page 7: 517 days for Ex. 2)",
            min_value=1.0,
            max_value=10000.0,
            value=(
                517.0
                if preset_choice.startswith("Example 2")
                else float(max(1.0, forecast_result.get("t_bt_days", 517.0)))
            ),
            step=1.0,
        )
    with col_tr3:
        connate_vol_override = st.number_input(
            "Contacted Connate Water Volume, Sw_avg*C*Ea*Vp (RB) (Table 11, WOR=10: 913,760 RB)",
            min_value=100.0,
            max_value=1.0e8,
            value=(
                913760.0
                if preset_choice.startswith("Example 2")
                else float(
                    sw_avg_bt_val
                    * forecast_result["C_bt"]
                    * forecast_result["Ea_bt"]
                    * vp_rb
                )
            ),
            step=1000.0,
            format="%.1f",
        )

    active_tracer_rows = [
        {**row, "m": float(m_multiple)} for row in active_preset["tracer_table"]
    ]
    w_tag_override = float(iw_bd * t_bt_override * bw + connate_vol_override)

    try:
        tracer_res = calculate_tracer_design(
            tracer_rows=active_tracer_rows,
            qi_bd=iw_bd,
            t_bt_days=t_bt_override,
            bw=bw,
            sw_avg_bt=sw_avg_bt_val,
            vp_rb=vp_rb,
            ea_bt=forecast_result["Ea_bt"],
            c_bt=forecast_result["C_bt"],
            override_w_tag_bbl=w_tag_override,
        )
        # Scan Page 8 Sensitivity Study: Breakthrough takes twice as long (t_bt = 644 days, W_tag = 1.948e6 RB for Ex. 2)
        sens_t_bt = 644.0 if preset_choice.startswith("Example 2") else (t_bt_override * 2.0)
        sens_w_tag = (
            1.948e6
            if preset_choice.startswith("Example 2")
            else float(iw_bd * sens_t_bt * bw + connate_vol_override)
        )
        sens_res = calculate_tracer_design(
            tracer_rows=active_tracer_rows,
            qi_bd=iw_bd,
            t_bt_days=sens_t_bt,
            bw=bw,
            sw_avg_bt=sw_avg_bt_val,
            vp_rb=vp_rb,
            ea_bt=forecast_result["Ea_bt"],
            c_bt=forecast_result["C_bt"],
            override_w_tag_bbl=sens_w_tag,
        )
    except Exception as e:
        st.warning(f"Tracer calculation error: {e}")
        tracer_res = {}
        sens_res = {}

    w_tag_val = tracer_res.get("W_tag_bbl", w_tag_override)
    tracer_df = pd.DataFrame(tracer_res.get("tracer_df", []))
    sens_df = pd.DataFrame(sens_res.get("tracer_df", []))

    st.latex(
        r"W_{tag} = q_i t_{bt} B_w + \overline{S}_{wbt} V_p E_a C, \quad "
        r"Q(\text{pCi}) = \frac{W_{tag} \, D \, m}{\zeta}, \quad "
        r"\zeta = \exp(-\lambda t_{bt}), \quad "
        r"C = \frac{Q}{q_i t_{bt}}, \quad "
        r"\%\text{MPC} = \frac{C}{\text{MPC}} \times 100"
    )
    st.info(
        f"**Injected Water Volume ($q_i t_{{bt}} B_w$):** `{iw_bd * t_bt_override * bw:,.0f} RB` | "
        f"**Contacted Connate Water Volume:** `{connate_vol_override:,.0f} RB` | "
        f"**Total Tagged Water Volume ($W_{{tag}}$):** `{w_tag_val:,.0f} RB`"
    )

    col_tb12, col_chart_tr = st.columns([3, 2])
    with col_tb12:
        st.markdown(
            "##### Table 10 & Table 12: Tracer Characteristics & Concentration Results (Scan Pages 23 & 25)"
        )
        if not tracer_df.empty:
            st.dataframe(tracer_df, use_container_width=True)
    with col_chart_tr:
        st.markdown(
            "##### Regulatory Safety Check: % of Maximum Permissible Concentration (%MPC)"
        )
        if not tracer_df.empty:
            tr_name_col = next(
                (c for c in ["Tracer", "tracer"] if c in tracer_df.columns),
                tracer_df.columns[0],
            )
            mpc_pct_col = next(
                (
                    c
                    for c in ["pct_MPC", "%MPC", "Percent_MPC"]
                    if c in tracer_df.columns
                ),
                None,
            )
            if mpc_pct_col:
                fig_tr = go.Figure()
                fig_tr.add_trace(
                    go.Bar(
                        x=tracer_df[tr_name_col],
                        y=tracer_df[mpc_pct_col],
                        text=[f"{v:.1f}%" for v in tracer_df[mpc_pct_col]],
                        textposition="outside",
                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"],
                        name="% MPC at Breakthrough",
                    )
                )
                fig_tr.add_hline(
                    y=100.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="100% MPC Regulatory Limit",
                )
                fig_tr.update_layout(
                    xaxis=dict(title="RADIOACTIVE TRACER ISOTOPE"),
                    yaxis=dict(
                        title="% OF MPC AT BREAKTHROUGH",
                        range=[
                            0,
                            max(110.0, float(tracer_df[mpc_pct_col].max()) * 1.2),
                        ],
                    ),
                    height=360,
                    template="plotly_white",
                )
                st.plotly_chart(fig_tr, use_container_width=True)

    if not sens_df.empty:
        st.markdown(
            "##### Scan Page 8 Sensitivity Study: Breakthrough Twice as Long ($t_{bt} = 644\\text{ days}$, $W_{tag} = 1.948 \\times 10^6\\text{ RB}$)"
        )
        st.dataframe(sens_df, use_container_width=True)