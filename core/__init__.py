"""
Core package for the WATER 3-D Single-Grid Waterflooding Simulator.
Tracks directly to Technical Service Report TSR 84-28 and SPE 13945.
"""

from .presets import (
    PATTERN_COEFFICIENTS,
    COVERAGE_CONSTANTS,
     get_example_1_preset,
    get_example_2_preset,
    TABLE_2_BENCHMARK,
    EXAMPLE_1_TABLE_7_MANUAL,
    EXAMPLE_1_CRAIG_CURVE,
)
from .rock_fluid import (
    calculate_permeability_variation,
    calculate_normalized_saturation,
    fit_rel_perm_power_law,
    build_rel_perm_table,
)
from .frac_flow import (
    calculate_fw_from_ratio,
    calculate_fw_from_sw,
    calculate_fw_derivative,
    solve_welge_breakthrough,
    solve_post_breakthrough_state,
    generate_fractional_flow_curve,
)
from .sweep_efficiency import (
    calculate_mobility_ratio,
    calculate_areal_sweep,
    calculate_coverage_parameter_y,
    solve_vertical_coverage,
)
from .forecaster import run_waterflood_forecast
from .tracer import calculate_tracer_design