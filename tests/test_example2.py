"""
tests/test_example2.py
Unit and integration tests for Example 2 (Staggered Line Drive & Radioactive Tracer Design)
validating against TSR 84-28 Scan Pages 6–8, 21–25 (Table 12), 38 (Fig. 10), and 39 (Fig. 11).
"""

import pytest

from core import (
    calculate_permeability_variation,
    calculate_tracer_design,
    fit_rel_perm_power_law,
    get_example_2_preset,
    run_waterflood_forecast,
    solve_welge_breakthrough,
)


@pytest.fixture
def ex2_preset():
    return get_example_2_preset()


def test_example2_volumetrics_and_printer_output_fig11(ex2_preset):
    """
    Validates Example 2 initial volumetrics and SLD recovery forecast against
    Figure 11 Printer Output (Scan Page 39):
    OOIP = 787,735.3846 STB, TFIL = 0.0000 years.
    """
    forecast = run_waterflood_forecast(
        vp_rb=ex2_preset["vp_rb"],
        swi=ex2_preset["swi"],
        sor=ex2_preset["sor"],
        sgi=ex2_preset["sgi"],
        kro_swi=ex2_preset["kro_swi"],
        mu_o=ex2_preset["mu_o_cp"],
        mu_w=ex2_preset["mu_w_cp"],
        bo=ex2_preset["bo"],
        bw=ex2_preset["bw"],
        iw_bd=ex2_preset["iw_bd"],
        pattern=ex2_preset["pattern"],
        v=ex2_preset["v_direct"],
        a=ex2_preset["a_direct"],
        b=ex2_preset["b_direct"],
        swbt=0.5649,
        fwbt=0.7477,
        sw_avg_bt=0.6206,
        wor_schedule=[0.1, 0.5, 1.0, 3.0, 3.8525, 5.0, 10.0],
        rel_perm_rows=ex2_preset["rel_perm_table"],
        after_bt_m_method="endpoint_krw",
    )

    assert forecast["OOIP_STB"] == pytest.approx(787735.3846, abs=1.0)
    assert forecast["t_fillup_years"] == pytest.approx(0.0, abs=1e-6)

    df = forecast["schedule_df"].set_index("WOR")

    # Check against Figure 11 Printer Output (Scan Page 39)
    # WOR = 0.1: EA = 0.6790, C = 0.2906, NP = 57,134.6, %REC = 7.2530
    assert df.loc[0.1, "Ea"] == pytest.approx(0.6790, abs=0.015)
    assert df.loc[0.1, "C"] == pytest.approx(0.2906, abs=0.015)
    assert df.loc[0.1, "OOIP_pct"] == pytest.approx(7.2530, abs=0.5)

    # WOR = 0.5: EA = 0.7292, C = 0.4157, %REC = 11.1423
    assert df.loc[0.5, "Ea"] == pytest.approx(0.7292, abs=0.015)
    assert df.loc[0.5, "C"] == pytest.approx(0.4157, abs=0.015)
    assert df.loc[0.5, "OOIP_pct"] == pytest.approx(11.1423, abs=0.5)

    # WOR = 1.0: EA = 0.7727, C = 0.5094, %REC = 14.4679
    assert df.loc[1.0, "Ea"] == pytest.approx(0.7727, abs=0.015)
    assert df.loc[1.0, "C"] == pytest.approx(0.5094, abs=0.015)
    assert df.loc[1.0, "OOIP_pct"] == pytest.approx(14.4679, abs=0.5)

    # WOR = 10.0: EA = 0.9703, C = 0.8573, %REC = 33.8722
    assert df.loc[10.0, "Ea"] == pytest.approx(0.9703, abs=0.015)
    assert df.loc[10.0, "C"] == pytest.approx(0.8573, abs=0.015)
    assert df.loc[10.0, "OOIP_pct"] == pytest.approx(33.8722, abs=1.5)


def test_example2_rel_perm_and_welge(ex2_preset):
    """
    Validates Example 2 relative permeability fit and Welge tangent breakthrough
    against Figure 10 (Scan Page 38) and Figure 11 (Scan Page 39).
    """
    fit_res = fit_rel_perm_power_law(
        rel_perm_rows=ex2_preset["rel_perm_table"],
        swi=ex2_preset["swi"],
        sor=ex2_preset["sor"],
    )
    assert fit_res["a"] == pytest.approx(2.68, abs=0.15)
    assert fit_res["b"] == pytest.approx(1.66, abs=0.10)

    bt_res = solve_welge_breakthrough(
        swi=ex2_preset["swi"],
        sor=ex2_preset["sor"],
        a=ex2_preset["a_direct"],
        b=ex2_preset["b_direct"],
        mu_o=ex2_preset["mu_o_cp"],
        mu_w=ex2_preset["mu_w_cp"],
    )
    assert bt_res["Swbt"] == pytest.approx(0.564, abs=0.005)
    assert bt_res["fwbt"] == pytest.approx(0.745, abs=0.015)
    assert bt_res["Sw_avg_bt"] == pytest.approx(0.621, abs=0.005)


def test_example2_permeability_variation(ex2_preset):
    """
    Validates that Subroutine KVA executes cleanly on Example 2 core_table
    and yields a valid Dykstra-Parsons V in (0.3, 0.8).
    """
    kva_res = calculate_permeability_variation(ex2_preset["core_table"])
    assert 0.3 <= kva_res["V"] <= 0.8
    assert ex2_preset["v_direct"] == pytest.approx(0.624, abs=1e-3)


def test_example2_tracer_design_table12(ex2_preset):
    """
    Validates radioactive tracer design module (core/tracer.py) against
    Table 12 (Scan Page 25) for HTO, Co-60, Co-57, and Ni-63.
    In Table 12, decay fraction zeta is evaluated at t_bt = 322 days (0.882 yrs)
    with W_tag = 455,000 bbl and injection volume = 517,000 bbl.
    """
    tracer_res = calculate_tracer_design(
        tracer_rows=ex2_preset["tracer_table"],
        qi_bd=ex2_preset["iw_bd"],
        t_bt_days=322.0,
        bw=ex2_preset["bw"],
        sw_avg_bt=ex2_preset["sw_avg_bt_direct"],
        vp_rb=ex2_preset["vp_rb"],
        ea_bt=0.858,
        c_bt=0.700,
        override_w_tag_bbl=455000.0,
        override_inj_vol_bbl=517000.0,
    )

    df = tracer_res["tracer_df"].set_index("Tracer")

    # 1. Check Decay Fractions (zeta) against Table 12 (Scan Page 25)
    assert df.loc["HTO", "zeta"] == pytest.approx(0.9515, abs=0.002)
    assert df.loc["Co-60", "zeta"] == pytest.approx(0.891, abs=0.002)
    assert df.loc["Co-57", "zeta"] == pytest.approx(0.439, abs=0.003)
    assert df.loc["Ni-63", "zeta"] == pytest.approx(0.994, abs=0.002)

    # 2. Check Total Activity Q(pCi) against Table 12 (Scan Page 25)
    assert df.loc["HTO", "Q_pCi"] == pytest.approx(23.91e12, rel=0.01)
    assert df.loc["Co-60", "Q_pCi"] == pytest.approx(121.2e9, rel=0.01)
    assert df.loc["Co-57", "Q_pCi"] == pytest.approx(246.0e9, rel=0.01)
    assert df.loc["Ni-63", "Q_pCi"] == pytest.approx(572.0e9, rel=0.01)

    # 3. Check Concentration C(pCi/bbl) and Regulatory %MPC against Table 12
    assert df.loc["HTO", "C_pCi_bbl"] == pytest.approx(46.25e6, rel=0.01)
    assert df.loc["HTO", "pct_MPC"] == pytest.approx(9.7, abs=0.2)
    assert df.loc["Co-60", "pct_MPC"] == pytest.approx(15.5, abs=0.2)
    assert df.loc["Co-57", "pct_MPC"] == pytest.approx(0.6, abs=0.1)
    assert df.loc["Ni-63", "pct_MPC"] == pytest.approx(23.2, abs=0.3)

    # All isotopes must be safely below 100% MPC
    assert (df["pct_MPC"] < 100.0).all()