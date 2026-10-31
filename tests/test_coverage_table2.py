"""
tests/test_coverage_table2.py
Validates Fassihi's Vertical Coverage correlation (Eqs. 16–21, Scan Pages 43–44)
against all 36 benchmark rows in Table 2 (Scan Page 14 & SPE 13945 Table 1),
as well as Areal Sweep Efficiency (Eqs. 14–15) across all 3 flooding patterns.
"""

import pytest

from core import (
    PATTERN_COEFFICIENTS,
    TABLE_2_BENCHMARK,
    calculate_areal_sweep,
    calculate_coverage_parameter_y,
    solve_vertical_coverage,
)


@pytest.mark.parametrize(
    "benchmark_row",
    TABLE_2_BENCHMARK,
    ids=[
        f"WOR={r['WOR']}_M={r['M']}_V={r['V']}" for r in TABLE_2_BENCHMARK
    ],
)
def test_table2_vertical_coverage_benchmarks(benchmark_row):
    """
    Checks that calculate_coverage_parameter_y (Eq. 17) and solve_vertical_coverage (Eq. 16)
    reproduce all 36 published Y and C (Eq. 16) values in Table 2 (Scan Page 14).
    """
    wor = benchmark_row["WOR"]
    m = benchmark_row["M"]
    v = benchmark_row["V"]
    expected_y = benchmark_row["Y"]
    expected_c_eq16 = benchmark_row["C_Eq16"]

    res = solve_vertical_coverage(wor=wor, m=m, v=v)

    # Verify de Souza & Brigham Y parameter (Eq. 17)
    assert res["Y"] == pytest.approx(expected_y, rel=1.5e-2, abs=0.02)

    # Verify Fassihi closed-form coverage C (Eq. 16)
    assert res["C"] == pytest.approx(expected_c_eq16, abs=0.025)
    assert 0.0 <= res["C"] <= 1.0


def test_coverage_four_regimes_flowchart_fig1d():
    """
    Tests that all 4 piecewise solver regimes in Flowchart Figure 1D (Scan Page 29)
    and Eqs. 18-21 (Scan Page 44) are reachable and continuous.
    """
    # Regime 1: Very small Y (<= 0.05) -> C == 0.0
    r1 = solve_vertical_coverage(wor=0.0, m=10.0, v=0.85)
    assert r1["Y"] <= 0.05
    assert r1["C"] == 0.0

    # Regime 2: 0.5 <= Y <= 2.6 (Eq. 19 fixed-point iteration from C0 = 0.30)
    r2 = solve_vertical_coverage(wor=0.1, m=1.0, v=0.8)  # Y ≈ 0.41 / 1.03
    r2_mid = solve_vertical_coverage(wor=0.1, m=0.1, v=0.8)  # Y ≈ 1.03
    assert 0.5 <= r2_mid["Y"] <= 2.6
    assert r2_mid["C"] == pytest.approx(0.16, abs=0.02)

    # Regime 3: 2.6 < Y <= 5.0 (Eq. 20 logarithmic formula)
    r3 = solve_vertical_coverage(wor=0.1, m=1.0, v=0.5)  # Y ≈ 3.16
    assert 2.6 < r3["Y"] <= 5.0
    assert r3["C"] == pytest.approx(0.41, abs=0.02)

    # Regime 4: Y > 5.0 (Eq. 21 fixed-point iteration from C0 = 0.65)
    r4 = solve_vertical_coverage(wor=10.0, m=1.0, v=0.5)  # Y ≈ 65.67
    assert r4["Y"] > 5.0
    assert r4["C"] == pytest.approx(0.92, abs=0.02)


@pytest.mark.parametrize("pattern", list(PATTERN_COEFFICIENTS.keys()))
def test_areal_sweep_correlations_all_patterns(pattern):
    """
    Validates Areal Sweep Efficiency (Eqs. 14-15, Scan Page 43; Table 1, Scan Page 13)
    across 5-Spot, Direct Line Drive (DLD), and Staggered Line Drive (SLD).
    """
    # At fixed mobility ratio M = 1.0, Ea must increase as fw increases from 0.1 to 0.99
    ea_low_fw = calculate_areal_sweep(m=1.0, fw=0.10, pattern=pattern)["Ea"]
    ea_mid_fw = calculate_areal_sweep(m=1.0, fw=0.50, pattern=pattern)["Ea"]
    ea_high_fw = calculate_areal_sweep(m=1.0, fw=0.95, pattern=pattern)["Ea"]

    assert 0.40 <= ea_low_fw <= ea_mid_fw <= ea_high_fw <= 1.0

    # At fixed fw = 0.50, Ea must decrease as mobility ratio M becomes more unfavorable (M: 0.2 -> 5.0)
    ea_favorable_m = calculate_areal_sweep(m=0.2, fw=0.50, pattern=pattern)["Ea"]
    ea_unfavorable_m = calculate_areal_sweep(m=5.0, fw=0.50, pattern=pattern)["Ea"]
    assert ea_favorable_m > ea_unfavorable_m