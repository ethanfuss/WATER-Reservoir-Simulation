# tests/test_arco_benchmarks.py
import numpy as np
import pytest
from core.rock_fluid import RockFluidProperties
from core.fractional_flow import FractionalFlowEngine
from core.sweep_efficiency import SweepEfficiencyEngine
from core.forecaster import WaterfloodForecaster
from core.tracer import RadioactiveTracerDesigner

def test_module1_example1_and_example2():
    # Example 1 Core Permeabilities (Table 6, p. 19)
    k_ex1 = [23.5, 18.2, 14.8, 12.2, 10.3, 8.7, 7.3, 6.0, 4.7, 3.3]
    res_v = RockFluidProperties.calculate_permeability_variation(k_ex1)
    assert round(float(res_v["V"]), 3) == 0.497  # Matches printer tape p. 36 (V=0.497)

    # Example 1 Rel-Perm Fit (Table 5, p. 18, Sw in [0.40, 0.60])
    sw_ex1 = [0.40, 0.45, 0.50, 0.55, 0.60]
    kro_ex1 = [0.204, 0.130, 0.077, 0.0418, 0.019]
    krw_ex1 = [0.066, 0.088, 0.110, 0.138, 0.168]
    fit_ex1 = RockFluidProperties.fit_relative_permeability(sw_ex1, kro_ex1, krw_ex1, swi=0.10, sor=0.30)
    assert round(float(fit_ex1["a"]), 3) == 1.202  # Matches printer tape p. 36
    assert round(float(fit_ex1["b"]), 3) == 1.796  # Matches printer tape p. 36

def test_module2_breakthrough_examples():
    # Example 1 Breakthrough (Figure 7, p. 35)
    ff1 = FractionalFlowEngine(a=1.202, b=1.796, swi=0.10, sor=0.30, mu_o=1.0, mu_w=0.5, bo=1.2, bw=1.0)
    bt1 = ff1.solve_breakthrough()
    assert round(float(bt1["Swbt"]), 3) == 0.463
    assert round(float(bt1["fwbt"]), 3) == 0.778
    assert round(float(bt1["Sw_bar_bt"]), 3) == 0.565

    # Example 2 Breakthrough & Post-BT (Figure 10, p. 38 & Table 11, p. 24)
    ff2 = FractionalFlowEngine(a=2.655, b=1.743, swi=0.40, sor=0.30, mu_o=2.5, mu_w=0.45, bo=1.3, bw=1.01)
    bt2 = ff2.solve_breakthrough()
    assert round(float(bt2["Swbt"]), 3) == 0.563
    assert round(float(bt2["fwbt"]), 3) == 0.734
    assert round(float(bt2["Sw_bar_bt"]), 4) == 0.6206

    st_wor5 = ff2.state_at_wor(5.0, bt2)
    assert round(float(st_wor5["Sw_bar"]), 4) == 0.6271
    st_wor10 = ff2.state_at_wor(10.0, bt2)
    assert round(float(st_wor10["Sw_bar"]), 3) == 0.644

def test_module3_coverage_table2_and_table11():
    # Check sample benchmarks from Table 2 (p. 14)
    c1 = SweepEfficiencyEngine.calculate_coverage(wor=0.1, m=0.1, v=0.3, method="HP41_exact")
    assert round(float(c1["Y"]), 2) == 16.24 and round(float(c1["C"]), 2) == 0.77

    c2 = SweepEfficiencyEngine.calculate_coverage(wor=1.0, m=1.0, v=0.5, method="HP41_exact")
    assert round(float(c2["Y"]), 2) == 8.84 and round(float(c2["C"]), 2) == 0.65

    c3 = SweepEfficiencyEngine.calculate_coverage(wor=50.0, m=10.0, v=0.8, method="HP41_exact")
    assert round(float(c3["Y"]), 2) == 5.86 and round(float(c3["C"]), 2) == 0.56

def test_module4_and_5_example2_tracer():
    vp_ex2 = RockFluidProperties.convert_acft_to_rb(220.0)
    forecaster = WaterfloodForecaster(
        vp_rb=vp_ex2, swi=0.40, sor=0.30, sgi=0.0, bo=1.3, bw=1.01,
        mu_o=2.5, mu_w=0.45, kro_soi=1.0, iw_bbd=500.0, v=0.624,
        a=2.655, b=1.743, pattern="staggered_line", krw_bt=0.085
    )
    df_ex2 = forecaster.run_forecast(
        wor_steps=[3.85, 5.0, 10.0],
        krw_manual_map={3.85: 0.085, 5.0: 0.093, 10.0: 0.10},
        solver_mode="HP41_exact",
        after_bt_mobility_mode="total"
    )
    # Verify Table 11 (p. 24)
    assert round(float(df_ex2["Ea"].to_numpy()[0]), 3) == 0.955
    assert round(float(df_ex2["C"].to_numpy()[0]), 3) == 0.731
    assert round(float(df_ex2["Swept_Water_MMrb"].to_numpy()[2]), 3) == 0.914

    # Verify Table 12 (p. 25)
    df_tracer = RadioactiveTracerDesigner.design_tracer_program(
        qi_bbd=500.0, bw=1.01, t_bt_days=322.0,
        swept_connate_water_rb=913760.0, detection_multiple_m=50.0,
        w_inj_override_rb=517000.0
    )
    hto = df_tracer[df_tracer["Tracer"] == "HTO"].iloc[0]
    assert round(float(hto["zeta_decay"]), 4) == 0.9515
    assert round(float(hto["Q_pCi"] / 1e12), 2) == 23.91
    assert round(float(hto["pct_MPC"]), 1) == 9.7

def test_all_table12_tracers():
    #Verifies all 4 isotopes against Table 12 (p. 25).
        df_tracer = RadioactiveTracerDesigner.design_tracer_program(
            qi_bbd=500.0,
            bw=1.01,
            t_bt_days=322.0,
            swept_connate_water_rb=913760.0,
            detection_multiple_m=50.0,
            w_inj_override_rb=517000.0
        )
        
        # Co-60
        co60 = df_tracer[df_tracer["Tracer"] == "Co-60"].iloc[0]
        assert round(float(co60["zeta_decay"]), 3) == 0.891
        assert round(float(co60["Q_pCi"]) / 1e9, 1) == 121.2

        # Co-57
        co57 = df_tracer[df_tracer["Tracer"] == "Co-57"].iloc[0]
        assert round(float(co57["zeta_decay"]), 3) == 0.439
        assert round(float(co57["Q_pCi"]) / 1e9, 0) == 246.0

        # Ni-63
        ni63 = df_tracer[df_tracer["Tracer"] == "Ni-63"].iloc[0]
        assert round(float(ni63["zeta_decay"]), 3) == 0.994
        assert round(float(ni63["Q_pCi"]) / 1e9, 0) == 572.0