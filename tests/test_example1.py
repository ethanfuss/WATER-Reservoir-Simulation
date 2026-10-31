"""
tests/test_example1.py
Unit and integration tests for Example 1 (Five-Spot General Waterflood)
validating against TSR 84-28 Scan Pages 5, 18–20, and 34–37.
"""

import numpy as np
import pytest

from core import (
    EXAMPLE_1_TABLE_7_MANUAL,
    build_rel_perm_table,
    calculate_normalized_saturation,
    calculate_permeability_variation,
    fit_rel_perm_power_law,
    get_example_1_preset,
    run_waterflood_forecast,
    solve_welge_breakthrough,
)


@pytest.fixture
def ex1_preset():
    return get_example_1_preset()


def test_example1_normalized_saturation_table5():
    """
    Validates Subroutine SWN against the handwritten Swn values at the bottom
    of Table 5 (Scan Page 18): Swn = (Sw - Swi) / (1 - Sw - Sor).
    """
    swi, sor = 0.10, 0.30
    expected_pairs = [
        (0.10, 0.0),
        (0.30, 0.5),
        (0.40, 1.0),
        (0.45, 1.4),
        (0.50, 2.0),
        (0.55, 3.0),
        (0.60, 5.0),
        (0.65, 11.0),
    ]
    for sw, expected_swn in expected_pairs:
        calc_swn = calculate_normalized_saturation(sw=sw, swi=swi, sor=sor)
        assert calc_swn == pytest.approx(expected_swn, rel=1e-4)


def test_example1_rel_perm_power_law_lin1(ex1_preset):
    """
    Validates Subroutine LIN1 (Eq. 2, Scan Page 41) on Table 5 (Scan Page 18).
    Must reproduce calculator printer output a = 1.202, b = 1.796 (Scan Page 36).
    """
    fit_res = fit_rel_perm_power_law(
        rel_perm_rows=ex1_preset["rel_perm_table"],
        swi=ex1_preset["swi"],
        sor=ex1_preset["sor"],
    )
    assert fit_res["n_points_used"] == 5
    assert fit_res["a"] == pytest.approx(1.202, abs=1e-2)
    assert fit_res["b"] == pytest.approx(1.796, abs=1e-2)
    assert fit_res["r_squared"] > 0.999


def test_example1_rel_perm_table5_fw(ex1_preset):
    """
    Validates Table 5 column 4 (fw_actual) against published Table 5 values (Scan Page 18).
    """
    df = build_rel_perm_table(
        rel_perm_rows=ex1_preset["rel_perm_table"],
        swi=ex1_preset["swi"],
        sor=ex1_preset["sor"],
        mu_o=ex1_preset["mu_o_cp"],
        mu_w=ex1_preset["mu_w_cp"],
        a=ex1_preset["a_direct"],
        b=ex1_preset["b_direct"],
    )
    expected_fw_actual = [0.0, 0.2729, 0.6168, 0.7533, 0.8571, 0.9250, 0.9665, 0.9901, 1.0000]
    np.testing.assert_allclose(df["fw_actual"].to_numpy(), expected_fw_actual, atol=1e-3)


def test_example1_permeability_variation_kva(ex1_preset):
    """
    Validates Subroutine KVA (Eq. 1, Scan Page 41) on Table 6 core data (Scan Page 19).
    """
    kva_res = calculate_permeability_variation(ex1_preset["core_table"])
    assert kva_res["V"] == pytest.approx(0.495, abs=0.01)
    assert kva_res["k50"] > kva_res["k84_1"] > 0.0
    assert len(kva_res["layer_df"]) == 10


def test_example1_welge_breakthrough_fig7(ex1_preset):
    """
    Validates Welge tangent breakthrough solver (Eqs. 4-10, Scan Pages 41-42)
    against Figure 7 (Scan Page 35) and Figure 8 (Scan Page 36):
    Swbt = 0.463, fwbt = 0.778, Sw_avg_bt = 0.565.
    """
    bt_res = solve_welge_breakthrough(
        swi=ex1_preset["swi"],
        sor=ex1_preset["sor"],
        a=ex1_preset["a_direct"],
        b=ex1_preset["b_direct"],
        mu_o=ex1_preset["mu_o_cp"],
        mu_w=ex1_preset["mu_w_cp"],
    )
    assert bt_res["Swbt"] == pytest.approx(0.463, abs=0.005)
    assert bt_res["fwbt"] == pytest.approx(0.778, abs=0.005)
    assert bt_res["Sw_avg_bt"] == pytest.approx(0.565, abs=0.005)
    assert bt_res["Slope2"] == pytest.approx(bt_res["Slope3"], abs=1e-3)


def test_example1_full_forecast_table7(ex1_preset):
    """
    Validates the full waterflood forecast (Eqs. 22-27, Scan Pages 44-45)
    against Example 1 volumetrics and Table 7 manual benchmark (Scan Page 20).
    """
    forecast = run_waterflood_forecast(
        vp_rb=ex1_preset["vp_rb"],
        swi=ex1_preset["swi"],
        sor=ex1_preset["sor"],
        sgi=ex1_preset["sgi"],
        kro_swi=ex1_preset["kro_swi"],
        mu_o=ex1_preset["mu_o_cp"],
        mu_w=ex1_preset["mu_w_cp"],
        bo=ex1_preset["bo"],
        bw=ex1_preset["bw"],
        iw_bd=ex1_preset["iw_bd"],
        pattern=ex1_preset["pattern"],
        v=ex1_preset["v_direct"],
        a=ex1_preset["a_direct"],
        b=ex1_preset["b_direct"],
        swbt=ex1_preset["swbt_direct"],
        fwbt=ex1_preset["fwbt_direct"],
        sw_avg_bt=ex1_preset["sw_avg_bt_direct"],
        wor_schedule=ex1_preset["wor_schedule"],
        rel_perm_rows=ex1_preset["rel_perm_table"],
        after_bt_m_method="endpoint_krw",
    )

    # Check initial volumetrics (Scan Page 52, Lines 53-75 & Scan Page 36, Fig. 8)
    assert forecast["OOIP_STB"] == pytest.approx(193950.0, abs=1.0)
    assert forecast["t_fillup_years"] == pytest.approx(0.474, abs=0.002)

    df = forecast["schedule_df"]
    assert len(df) == len(EXAMPLE_1_TABLE_7_MANUAL)

    # Check monotonic physical behavior across WOR steps
    assert df["Np_STB"].is_monotonic_increasing
    assert df["Wp_STB"].is_monotonic_increasing
    assert df["Time_years"].is_monotonic_increasing
    assert df["C"].is_monotonic_increasing

    # Compare against Table 7 (Scan Page 20)
    for idx, manual_row in enumerate(EXAMPLE_1_TABLE_7_MANUAL):
        sim_row = df.iloc[idx]
        assert sim_row["WOR"] == pytest.approx(manual_row["WOR"], rel=1e-5)
        assert sim_row["fw"] == pytest.approx(manual_row["fw"], abs=2e-3)
        assert sim_row["Sw"] == pytest.approx(manual_row["Sw"], abs=0.01)
        assert sim_row["M"] == pytest.approx(manual_row["M"], abs=0.05)
        assert sim_row["Ea"] == pytest.approx(manual_row["Ea"], abs=0.04)
        assert sim_row["C"] == pytest.approx(manual_row["C"], abs=0.04)
        assert sim_row["Fwp"] == pytest.approx(manual_row["Fwp"], rel=1e-3)
        assert sim_row["OOIP_pct"] == pytest.approx(manual_row["OOIP_pct"], abs=2.5)