# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.rock_fluid import RockFluidProperties
from core.fractional_flow import FractionalFlowEngine
from core.sweep_efficiency import SweepEfficiencyEngine
from core.forecaster import WaterfloodForecaster
from core.tracer import RadioactiveTracerDesigner

st.set_page_config(page_title="ARCO WATER Simulator (TSR 84-28)", layout="wide")
st.title("ARCO WATER — Analytical 3D Waterflood & Tracer Simulator")
st.caption("Faithful Python implementation of M.R. Fassihi & W.J. O'Brien (ARCO TSR 84-28 / SPE 13945)")

# --- Sidebar Preset & Controls ---
st.sidebar.header("1. Benchmark Preset & Solver Mode")
preset = st.sidebar.selectbox(
    "Load Benchmark Case",
    ["Example 1: Five-Spot Waterflood (p. 5)", "Example 2: SLD & Tracer Design (p. 6)", "Custom Reservoir"]
)
solver_mode = st.sidebar.radio(
    "Numerical Solver Mode",
    ["HP41_exact", "continuous"],
    help="'HP41_exact' uses the 1984 4-regime piecewise coverage rules (p. 44). 'continuous' uses SciPy Brent root-finding."
)

if "Example 1" in preset:
    def_pattern, def_vp, def_swi, def_sor, def_sgi = "5-spot", 310320.0, 0.10, 0.30, 0.15
    def_bo, def_bw, def_muo, def_muw, def_iw = 1.20, 1.00, 1.0, 0.5, 269.1
    def_v, def_a, def_b, def_krw_bt = 0.4975, 1.202, 1.796, 0.152
    def_wors = "0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0"
    mob_mode = "water_only"
elif "Example 2" in preset:
    def_pattern, def_vp, def_swi, def_sor, def_sgi = "staggered_line", 1706760.0, 0.40, 0.30, 0.00
    def_bo, def_bw, def_muo, def_muw, def_iw = 1.30, 1.01, 2.5, 0.45, 500.0
    def_v, def_a, def_b, def_krw_bt = 0.624, 2.655, 1.743, 0.085
    def_wors = "3.85, 5.0, 10.0, 25.0, 50.0"
    mob_mode = "total"
else:
    def_pattern, def_vp, def_swi, def_sor, def_sgi = "5-spot", 500000.0, 0.20, 0.30, 0.05
    def_bo, def_bw, def_muo, def_muw, def_iw = 1.25, 1.02, 1.5, 0.5, 400.0
    def_v, def_a, def_b, def_krw_bt = 0.55, 1.50, 1.80, 0.12
    def_wors = "0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0"
    mob_mode = "total"

st.sidebar.header("2. Rock & Fluid Inputs")
pattern = st.sidebar.selectbox("Flooding Pattern", ["5-spot", "direct_line", "staggered_line"],
                               index=["5-spot", "direct_line", "staggered_line"].index(def_pattern))
vp_rb = st.sidebar.number_input("Pore Volume Vp (rb)", value=def_vp, step=10000.0)
swi = st.sidebar.slider("Irreducible Water Swi", 0.05, 0.50, def_swi, 0.01)
sor = st.sidebar.slider("Residual Oil Sor", 0.10, 0.50, def_sor, 0.01)
sgi = st.sidebar.slider("Initial Gas Saturation Sgi", 0.00, 0.30, def_sgi, 0.01)
mu_o = st.sidebar.number_input("Oil Viscosity μo (cp)", value=def_muo, step=0.1)
mu_w = st.sidebar.number_input("Water Viscosity μw (cp)", value=def_muw, step=0.05)
bo = st.sidebar.number_input("Oil FVF Bo (rb/STB)", value=def_bo, step=0.05)
bw = st.sidebar.number_input("Water FVF Bw (rb/STB)", value=def_bw, step=0.01)
iw_bbd = st.sidebar.number_input("Water Injection Rate iw (B/D)", value=def_iw, step=10.0)

st.sidebar.header("3. Heterogeneity & Rel-Perm Constants")
v_val = st.sidebar.slider("Dykstra-Parsons V", 0.30, 0.80, def_v, 0.001)
a_val = st.sidebar.number_input("Rel-Perm Constant 'a'", value=def_a, step=0.01)
b_val = st.sidebar.number_input("Rel-Perm Exponent 'b'", value=def_b, step=0.01)
krw_bt = st.sidebar.number_input("krw at Breakthrough Sw_bar", value=def_krw_bt, step=0.005)
wor_str = st.sidebar.text_input("Target WOR Steps (comma-separated)", value=def_wors)
wor_list = [float(x.strip()) for x in wor_str.split(",") if x.strip()]

# --- Execute Core Pipeline ---
init_res = RockFluidProperties.initialize_reservoir(vp_rb, swi, sor, sgi, bo, bw, iw_bbd)
ff_engine = FractionalFlowEngine(a_val, b_val, swi, sor, mu_o, mu_w, bo, bw)
bt_res = ff_engine.solve_breakthrough(method="HP41_exact" if solver_mode == "HP41_exact" else "continuous")

krw_map = {3.85: 0.085, 5.0: 0.093, 10.0: 0.10} if "Example 2" in preset else None
forecaster = WaterfloodForecaster(
    vp_rb=vp_rb, swi=swi, sor=sor, sgi=sgi, bo=bo, bw=bw,
    mu_o=mu_o, mu_w=mu_w, kro_soi=1.0, iw_bbd=iw_bbd, v=v_val,
    a=a_val, b=b_val, pattern=pattern, krw_bt=krw_bt
)
df_forecast = forecaster.run_forecast(wor_list, krw_manual_map=krw_map,
                                      solver_mode=solver_mode, after_bt_mobility_mode=mob_mode)

# --- Top KPI Banner ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Original Oil in Place (OOIP)", f"{init_res['OOIP_STB']:,.0f} STB")
c2.metric("Fill-Up Time (t_fil)", f"{init_res['t_fillup_years']:.3f} yrs")
c3.metric("Breakthrough Front (Swbt)", f"{bt_res['Swbt']:.3f}")
c4.metric("Avg BT Saturation (Sw_bar_bt)", f"{bt_res['Sw_bar_bt']:.3f}")
c5.metric("Breakthrough fwbt", f"{bt_res['fwbt']:.3f}")

# --- Tabs for Visuals ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fractional Flow & Welge Tangent",
    "2. 3D Sweep Efficiency Surfaces (Ea & C)",
    "3. Production Forecast & Ledger",
    "4. Radioactive Tracer Design (Table 12)"
])

with tab1:
    sw_vec = np.linspace(swi + 1e-4, 1.0 - sor - 1e-4, 300)
    fw_vec = ff_engine.fractional_flow(sw_vec)
    # Welge Tangent Line from (Swi, 0) through (Swbt, fwbt) to (Sw_bar_bt, 1.0)
    fig_ff = go.Figure()
    fig_ff.add_trace(go.Scatter(x=sw_vec, y=fw_vec, name="Fractional Flow fw(Sw)", line=dict(width=3)))
    fig_ff.add_trace(go.Scatter(
        x=[swi, bt_res["Swbt"], bt_res["Sw_bar_bt"]],
        y=[0.0, bt_res["fwbt"], 1.0],
        mode="lines+markers",
        name="Welge Breakthrough Tangent",
        line=dict(dash="dash", color="red")
    ))
    fig_ff.update_layout(title="Buckley-Leverett Fractional Flow & Welge Tangent Construction",
                         xaxis_title="Water Saturation (Sw)", yaxis_title="Fractional Flow (fw)",
                         xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig_ff, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    m_grid = np.linspace(0.1, 10.0, 40)
    fw_grid = np.linspace(0.0, 1.0, 40)
    M_mesh, FW_mesh = np.meshgrid(m_grid, fw_grid)
    EA_mesh = np.vectorize(lambda m, f: SweepEfficiencyEngine.calculate_areal_sweep(m, f, pattern))(M_mesh, FW_mesh)

    fig_ea = go.Figure(data=[go.Surface(z=EA_mesh, x=M_mesh, y=FW_mesh, colorscale="Viridis")])
    fig_ea.update_layout(title=f"Areal Sweep Efficiency (Ea) — {pattern}",
                         scene=dict(xaxis_title="Mobility Ratio M", yaxis_title="Fractional Flow fw", zaxis_title="Ea"))
    col_a.plotly_chart(fig_ea, use_container_width=True)

    v_grid = np.linspace(0.3, 0.8, 35)
    wor_grid = np.linspace(0.1, 50.0, 35)
    V_mesh, WOR_mesh = np.meshgrid(v_grid, wor_grid)
    C_mesh = np.vectorize(lambda w, v: SweepEfficiencyEngine.calculate_coverage(w, 1.0, v, solver_mode)["C"])(WOR_mesh, V_mesh)

    fig_c = go.Figure(data=[go.Surface(z=C_mesh, x=V_mesh, y=WOR_mesh, colorscale="Plasma")])
    fig_c.update_layout(title=f"Vertical Coverage (C) at M=1.0 ({solver_mode})",
                        scene=dict(xaxis_title="Permeability Variation V", yaxis_title="WOR", zaxis_title="Coverage C"))
    col_b.plotly_chart(fig_c, use_container_width=True)

with tab3:
    fig_prod = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prod.add_trace(go.Scatter(x=df_forecast["Time_yr"], y=df_forecast["Np_STB"],
                                  mode="lines+markers", name="Cumulative Oil Np (STB)"), secondary_y=False)
    fig_prod.add_trace(go.Scatter(x=df_forecast["Time_yr"], y=df_forecast["WOR"],
                                  mode="lines+markers", name="Producing WOR", line=dict(dash="dot")), secondary_y=True)
    fig_prod.update_layout(title="Waterflood Recovery & Producing WOR vs. Project Time (Years)",
                           xaxis_title="Project Time (Years)")
    fig_prod.update_yaxes(title_text="Cumulative Oil Recovery Np (STB)", secondary_y=False)
    fig_prod.update_yaxes(title_text="Water-Oil Ratio (WOR)", secondary_y=True)
    st.plotly_chart(fig_prod, use_container_width=True)
    st.dataframe(df_forecast, use_container_width=True)

with tab4:
    st.subheader("Radioactive Tracer Injection Design & Regulatory Safety Check (Pages 6–8, Tables 10–12)")
    t_col1, t_col2, t_col3 = st.columns(3)
    t_bt_days = t_col1.number_input("Tracer Breakthrough Time t_bt (days)", value=322.0, step=10.0)
    swept_water_rb = t_col2.number_input("Contacted Connate Water Volume (rb)",
                                         value=float(df_forecast.iloc[-1]["Swept_Water_MMrb"] * 1e6), step=10000.0)
    m_mult = t_col3.number_input("Detection Limit Multiple (m)", value=50.0, step=5.0)

    df_tracer = RadioactiveTracerDesigner.design_tracer_program(
        qi_bbd=iw_bbd, bw=bw, t_bt_days=t_bt_days,
        swept_connate_water_rb=swept_water_rb, detection_multiple_m=m_mult
    )
    st.dataframe(df_tracer, use_container_width=True)

    st.markdown("#### Delayed Breakthrough Sensitivity (2x Breakthrough Time — Page 8)")
    df_sens = RadioactiveTracerDesigner.evaluate_late_breakthrough_sensitivity(
        df_tracer, qi_bbd=iw_bbd, bw=bw, new_t_bt_days=t_bt_days * 2.0,
        swept_connate_water_rb=swept_water_rb, base_w_tag_rb=float(df_tracer["W_tag_rb"].iloc[0])
    )
    st.dataframe(df_sens, use_container_width=True)