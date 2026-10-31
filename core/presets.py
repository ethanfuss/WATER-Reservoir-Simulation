"""
core/presets.py
Preset input parameters and validation tables for Example 1 and Example 2.
All reservoir parameters are loaded dynamically into the UI so nothing is hardcoded
in the mathematical engines.

Scan References:
- Scan Page 5 (e1input.jpg): Example 1 reservoir properties
- Scan Page 6: Example 2 reservoir properties & tracer equations
- Scan Page 13 (Table 1) & SPE 13945 Page 605 (Table 2): Areal sweep regression constants
- Scan Page 14 (Table 2): Coverage correlation validation table
- Scan Page 18 (Table 5, table5.jpg): Relative permeability data for Example 1
- Scan Page 19 (Table 6, table6.jpg): Core permeabilities for Example 1
- Scan Page 20 (Table 7, table7.jpg): Hand-calculated recovery table for Example 1
- Scan Page 25 (Table 12): Tracer concentration data for Example 2
- Scan Page 37 (Figure 9, fig9.jpg): Comparison of Craig's Data, Manual, and Calculator Results
- Scan Page 43 (Eq. 16): Vertical coverage regression constants a1, a2, a3
"""

from typing import Dict, Any, List

# Scan Page 13, Table 1: Coefficients in Areal Sweep Efficiency Correlations (Eq. 15, Scan Page 43)
# Valid for 0 < M < 10 before and after breakthrough (Scan Page 4)
PATTERN_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "5-Spot": {
        "a1": -0.2062,
        "a2": -0.0712,
        "a3": -0.5110,
        "a4": 0.3048,
        "a5": 0.1230,
        "a6": 0.4394,
    },
    "Direct Line Drive (DLD)": {
        "a1": -0.3014,
        "a2": -0.1568,
        "a3": -0.9402,
        "a4": 0.3714,
        "a5": -0.0865,
        "a6": 0.8805,
    },
    "Staggered Line Drive (SLD)": {
        "a1": -0.2077,
        "a2": -0.1059,
        "a3": -0.3526,
        "a4": 0.2608,
        "a5": 0.2444,
        "a6": 0.3158,
    },
}

# Scan Page 43, Eq. 16: Constants in Coverage Correlation Y = a1 * C^a2 * (1 - C)^a3
COVERAGE_CONSTANTS: Dict[str, float] = {
    "a1": 3.334088568,
    "a2": 0.7737348199,
    "a3": -1.225859406,
}

# Default WOR schedule specified on Scan Page 4 ("Calculation of Vertical Coverage")
# and Scan Page 20 (Table 7, table7.jpg)
DEFAULT_WOR_SCHEDULE: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]


def get_example_1_preset() -> Dict[str, Any]:
    """
    Returns all input parameters and tables for Example 1: A general waterflooding problem.
    Tracks to Scan Page 5 (e1input.jpg), Scan Page 18 (Table 5, table5.jpg),
    and Scan Page 19 (Table 6, table6.jpg).
    """
    return {
        "name": "Example 1 (Five-Spot General Waterflood)",
        "pattern": "5-Spot",
        "k_nominal_md": 10.0,       # Scan Page 5 (e1input.jpg)
        "vp_rb": 310320.0,          # Scan Page 5 (e1input.jpg)
        "swi": 0.10,                # Scan Page 18 (Table 5, first row where krw=0)
        "sor": 0.30,                # Scan Page 5 (e1input.jpg) & Scan Page 18 (Table 5)
        "sgi": 0.15,                # Scan Page 5 (e1input.jpg)
        "kro_swi": 1.000,           # Scan Page 18 (Table 5, kro at Sw=0.10)
        "mu_o_cp": 1.0,             # Scan Page 5 (e1input.jpg)
        "mu_w_cp": 0.5,             # Scan Page 5 (e1input.jpg)
        "bo": 1.2,                  # Scan Page 5 (e1input.jpg)
        "bw": 1.0,                  # Scan Page 5 (e1input.jpg)
        "iw_bd": 269.1,             # Scan Page 5 (e1input.jpg)
        "v_mode": "Estimate from Core Layers (Subroutine KVA)",
        "v_direct": 0.50,
        "ab_mode": "Estimate from Rel-Perm Table (Suboutines SWN & LIN1)",
        "a_direct": 1.202,
        "b_direct": 1.796,
        "swbt_mode": "Compute via Welge Tangent (Subroutine SWBT)",
        "swbt_direct": 0.463,       # Scan Page 35 (Figure 7, fig7.jpg)
        "sw_avg_bt_direct": 0.565,  # Scan Page 35 (Figure 7, fig7.jpg)
        "fwbt_direct": 0.778,       # Scan Page 35 (Figure 7, fig7.jpg)
        "wor_schedule": DEFAULT_WOR_SCHEDULE.copy(),
        # Scan Page 18, Table 5 (table5.jpg): Relative Permeability Data for Example 1
        # Note: The pencil bracket in table5.jpg spans Sw = 0.40 to 0.60, which yields
        # the exact calculator regression constants a = 1.202, b = 1.796.
        "rel_perm_table": [
            {"Sw": 0.10, "kro": 1.000, "krw": 0.000, "fit_include": False},
            {"Sw": 0.30, "kro": 0.373, "krw": 0.070, "fit_include": False},
            {"Sw": 0.40, "kro": 0.210, "krw": 0.169, "fit_include": True},
            {"Sw": 0.45, "kro": 0.148, "krw": 0.226, "fit_include": True},
            {"Sw": 0.50, "kro": 0.100, "krw": 0.300, "fit_include": True},
            {"Sw": 0.55, "kro": 0.061, "krw": 0.376, "fit_include": True},
            {"Sw": 0.60, "kro": 0.033, "krw": 0.476, "fit_include": True},
            {"Sw": 0.65, "kro": 0.012, "krw": 0.600, "fit_include": False},
            {"Sw": 0.70, "kro": 0.000, "krw": 0.740, "fit_include": False},
        ],
        # Scan Page 19, Table 6 (table6.jpg): Core Permeabilities in Example 1
        "core_table": [
            {"Layer": 1, "k_md": 31.5, "Thickness_ft": 5.0},
            {"Layer": 2, "k_md": 20.5, "Thickness_ft": 5.0},
            {"Layer": 3, "k_md": 16.0, "Thickness_ft": 5.0},
            {"Layer": 4, "k_md": 13.1, "Thickness_ft": 5.0},
            {"Layer": 5, "k_md": 10.9, "Thickness_ft": 5.0},
            {"Layer": 6, "k_md": 8.2,  "Thickness_ft": 5.0},
            {"Layer": 7, "k_md": 7.7,  "Thickness_ft": 5.0},
            {"Layer": 8, "k_md": 6.3,  "Thickness_ft": 5.0},
            {"Layer": 9, "k_md": 4.9,  "Thickness_ft": 5.0},
            {"Layer": 10, "k_md": 3.2, "Thickness_ft": 5.0},
        ],
        "tracer_table": [
            {"Tracer": "HTO",   "t_half_days": 4475.0,  "D_pCi_bbl": 1.0e6,  "m": 50.0, "MPC_pCi_bbl": 4.768e8},
            {"Tracer": "Co-60", "t_half_days": 1920.0,  "D_pCi_bbl": 4750.0, "m": 50.0, "MPC_pCi_bbl": 1.513e6},
            {"Tracer": "Co-57", "t_half_days": 270.0,   "D_pCi_bbl": 4750.0, "m": 50.0, "MPC_pCi_bbl": 7.930e7},
            {"Tracer": "Ni-63", "t_half_days": 36500.0, "D_pCi_bbl": 25000.0,"m": 50.0, "MPC_pCi_bbl": 4.770e6},
        ],
    }


def get_example_2_preset() -> Dict[str, Any]:
    """
    Returns all input parameters and tables for Example 2: Design of Radioactive Tracer Injection.
    Tracks to Scan Page 6, Scan Page 25 (Table 12), Scan Page 34 (Figure 6, fig6.jpg),
    and Scan Page 38 (Figure 10).
    Note: Vp = 220 Ac-ft = 220 * 7758 = 1,706,760 RB (Scan Page 6).
    """
    return {
        "name": "Example 2 (Staggered Line Drive & Tracer Design)",
        "pattern": "Staggered Line Drive (SLD)",
        "k_nominal_md": 15.0,
        "vp_rb": 220.0 * 7758.0,    # Scan Page 6: 220 Ac-ft * 7758 bbl/Ac-ft = 1,706,760 rb
        "swi": 0.40,                # Scan Page 6
        "sor": 0.30,                # Scan Page 6
        "sgi": 0.00,                # Scan Page 6
        "kro_swi": 0.800,           # Scan Page 6 / Table 8
        "mu_o_cp": 2.5,             # Scan Page 6
        "mu_w_cp": 0.45,            # Scan Page 6
        "bo": 1.3,                  # Scan Page 6
        "bw": 1.01,                 # Scan Page 6
        "iw_bd": 500.0,             # Scan Page 6
        "v_mode": "Estimate from Core Layers (Subroutine KVA)",
        "v_direct": 0.624,          # Scan Page 7
        "ab_mode": "Estimate from Rel-Perm Table (Suboutines SWN & LIN1)",
        "a_direct": 2.680,
        "b_direct": 1.660,
        "swbt_mode": "Compute via Welge Tangent (Subroutine SWBT)",
        "swbt_direct": 0.563,       # Scan Page 38 (Figure 10)
        "sw_avg_bt_direct": 0.621,  # Scan Page 38 (Figure 10)
        "fwbt_direct": 0.734,       # Scan Page 38 (Figure 10)
        "wor_schedule": DEFAULT_WOR_SCHEDULE.copy(),
        # Scan Page 21 (Table 8) & Scan Page 34 (Figure 6) & Scan Page 38 (Figure 10)
        "rel_perm_table": [
            {"Sw": 0.40, "kro": 0.800, "krw": 0.000,  "fit_include": False},
            {"Sw": 0.45, "kro": 0.590, "krw": 0.015,  "fit_include": False},
            {"Sw": 0.50, "kro": 0.390, "krw": 0.046,  "fit_include": True},
            {"Sw": 0.55, "kro": 0.232, "krw": 0.087,  "fit_include": True},
            {"Sw": 0.60, "kro": 0.105, "krw": 0.138,  "fit_include": True},
            {"Sw": 0.65, "kro": 0.032, "krw": 0.174,  "fit_include": True},
            {"Sw": 0.685,"kro": 0.005, "krw": 0.250,  "fit_include": True},
            {"Sw": 0.70, "kro": 0.000, "krw": 0.300,  "fit_include": False},
        ],
        # Scan Page 22 (Table 9): Core Permeabilities yielding V = 0.624 (Scan Page 7)
        "core_table": [
            {"Layer": 1,  "k_md": 82.0, "Thickness_ft": 2.0},
            {"Layer": 2,  "k_md": 45.0, "Thickness_ft": 2.0},
            {"Layer": 3,  "k_md": 29.0, "Thickness_ft": 2.0},
            {"Layer": 4,  "k_md": 20.0, "Thickness_ft": 2.0},
            {"Layer": 5,  "k_md": 14.5, "Thickness_ft": 2.0},
            {"Layer": 6,  "k_md": 10.5, "Thickness_ft": 2.0},
            {"Layer": 7,  "k_md": 7.6,  "Thickness_ft": 2.0},
            {"Layer": 8,  "k_md": 5.3,  "Thickness_ft": 2.0},
            {"Layer": 9,  "k_md": 3.4,  "Thickness_ft": 2.0},
            {"Layer": 10, "k_md": 1.8,  "Thickness_ft": 2.0},
        ],
        # Scan Page 23 (Table 10) & Scan Page 25 (Table 12)
        "tracer_table": [
            {"Tracer": "HTO",   "t_half_days": 4475.0,  "D_pCi_bbl": 1.0e6,  "m": 50.0, "MPC_pCi_bbl": 4.768e8},
            {"Tracer": "Co-60", "t_half_days": 1920.0,  "D_pCi_bbl": 4750.0, "m": 50.0, "MPC_pCi_bbl": 1.513e6},
            {"Tracer": "Co-57", "t_half_days": 270.0,   "D_pCi_bbl": 4750.0, "m": 50.0, "MPC_pCi_bbl": 7.930e7},
            {"Tracer": "Ni-63", "t_half_days": 36500.0, "D_pCi_bbl": 25000.0,"m": 50.0, "MPC_pCi_bbl": 4.770e6},
        ],
    }


# Scan Page 20, Table 7 (table7.jpg): Hand-Calculated Oil Recovery Table for Example 1
EXAMPLE_1_TABLE_7_MANUAL: List[Dict[str, float]] = [
    {"WOR": 0.1,   "fw": 0.0909, "Sw": 0.563, "dSo": 0.313, "M": 0.80, "Ea": 0.745, "C": 0.357, "dSo_Ea_C": 0.083, "OOIP_pct": 11.1, "Fwp": 0.12},
    {"WOR": 0.2,   "fw": 0.1670, "Sw": 0.563, "dSo": 0.313, "M": 0.80, "Ea": 0.760, "C": 0.435, "dSo_Ea_C": 0.103, "OOIP_pct": 13.8, "Fwp": 0.24},
    {"WOR": 0.5,   "fw": 0.3330, "Sw": 0.563, "dSo": 0.313, "M": 0.80, "Ea": 0.800, "C": 0.567, "dSo_Ea_C": 0.142, "OOIP_pct": 18.9, "Fwp": 0.60},
    {"WOR": 1.0,   "fw": 0.5000, "Sw": 0.563, "dSo": 0.313, "M": 0.80, "Ea": 0.845, "C": 0.677, "dSo_Ea_C": 0.179, "OOIP_pct": 23.9, "Fwp": 1.20},
    {"WOR": 2.0,   "fw": 0.6670, "Sw": 0.563, "dSo": 0.313, "M": 0.80, "Ea": 0.900, "C": 0.782, "dSo_Ea_C": 0.220, "OOIP_pct": 29.4, "Fwp": 2.40},
    {"WOR": 5.0,   "fw": 0.8330, "Sw": 0.578, "dSo": 0.328, "M": 0.84, "Ea": 0.970, "C": 0.880, "dSo_Ea_C": 0.280, "OOIP_pct": 37.3, "Fwp": 6.00},
    {"WOR": 10.0,  "fw": 0.9090, "Sw": 0.611, "dSo": 0.361, "M": 0.99, "Ea": 0.985, "C": 0.922, "dSo_Ea_C": 0.328, "OOIP_pct": 43.7, "Fwp": 12.0},
    {"WOR": 25.0,  "fw": 0.9610, "Sw": 0.647, "dSo": 0.397, "M": 1.18, "Ea": 0.995, "C": 0.959, "dSo_Ea_C": 0.379, "OOIP_pct": 50.5, "Fwp": 30.0},
    {"WOR": 50.0,  "fw": 0.9800, "Sw": 0.666, "dSo": 0.416, "M": 1.28, "Ea": 1.000, "C": 0.976, "dSo_Ea_C": 0.406, "OOIP_pct": 54.1, "Fwp": 60.0},
    {"WOR": 100.0, "fw": 0.9900, "Sw": 0.681, "dSo": 0.431, "M": 1.35, "Ea": 1.000, "C": 0.987, "dSo_Ea_C": 0.425, "OOIP_pct": 56.7, "Fwp": 120.0},
]

# Scan Page 37, Figure 9 (fig9.jpg): Craig's Monograph Benchmark Points (Triangles in Fig. 9)
EXAMPLE_1_CRAIG_CURVE: List[Dict[str, float]] = [
    {"OOIP_pct": 11.1, "WOR": 0.12},
    {"OOIP_pct": 30.1, "WOR": 1.75},
    {"OOIP_pct": 40.0, "WOR": 5.00},
    {"OOIP_pct": 45.3, "WOR": 9.80},
    {"OOIP_pct": 50.4, "WOR": 16.9},
    {"OOIP_pct": 54.5, "WOR": 29.8},
    {"OOIP_pct": 56.1, "WOR": 39.8},
]

# Scan Page 14, Table 2: Comparison Between Coverage Values (Dykstra-Parsons Ref. 3 vs Eq. 16)
TABLE_2_BENCHMARK: List[Dict[str, float]] = [
    {"WOR": 0.1,  "M": 0.1,  "V": 0.3, "Y": 16.24,   "C_Ref3": 0.74, "C_Eq16": 0.77},
    {"WOR": 0.1,  "M": 0.1,  "V": 0.5, "Y": 6.57,    "C_Ref3": 0.54, "C_Eq16": 0.59},
    {"WOR": 0.1,  "M": 0.1,  "V": 0.8, "Y": 1.03,    "C_Ref3": 0.19, "C_Eq16": 0.16},
    {"WOR": 0.1,  "M": 1.0,  "V": 0.3, "Y": 8.52,    "C_Ref3": 0.58, "C_Eq16": 0.64},
    {"WOR": 0.1,  "M": 1.0,  "V": 0.5, "Y": 3.16,    "C_Ref3": 0.32, "C_Eq16": 0.41},
    {"WOR": 0.1,  "M": 1.0,  "V": 0.8, "Y": 0.41,    "C_Ref3": 0.05, "C_Eq16": 0.06},
    {"WOR": 0.1,  "M": 10.0, "V": 0.3, "Y": 1.48,    "C_Ref3": 0.28, "C_Eq16": 0.23},
    {"WOR": 0.1,  "M": 10.0, "V": 0.5, "Y": 0.51,    "C_Ref3": 0.10, "C_Eq16": 0.08},
    {"WOR": 0.1,  "M": 10.0, "V": 0.8, "Y": 0.06,    "C_Ref3": 0.02, "C_Eq16": 0.01},
    {"WOR": 1.0,  "M": 0.1,  "V": 0.3, "Y": 45.46,   "C_Ref3": 0.90, "C_Eq16": 0.89},
    {"WOR": 1.0,  "M": 0.1,  "V": 0.5, "Y": 18.40,   "C_Ref3": 0.79, "C_Eq16": 0.79},
    {"WOR": 1.0,  "M": 0.1,  "V": 0.8, "Y": 2.90,    "C_Ref3": 0.45, "C_Eq16": 0.39},
    {"WOR": 1.0,  "M": 1.0,  "V": 0.3, "Y": 23.86,   "C_Ref3": 0.83, "C_Eq16": 0.82},
    {"WOR": 1.0,  "M": 1.0,  "V": 0.5, "Y": 8.84,    "C_Ref3": 0.65, "C_Eq16": 0.65},
    {"WOR": 1.0,  "M": 1.0,  "V": 0.8, "Y": 1.15,    "C_Ref3": 0.23, "C_Eq16": 0.18},
    {"WOR": 1.0,  "M": 10.0, "V": 0.3, "Y": 4.15,    "C_Ref3": 0.49, "C_Eq16": 0.48},
    {"WOR": 1.0,  "M": 10.0, "V": 0.5, "Y": 1.43,    "C_Ref3": 0.25, "C_Eq16": 0.22},
    {"WOR": 1.0,  "M": 10.0, "V": 0.8, "Y": 0.17,    "C_Ref3": 0.04, "C_Eq16": 0.02},
    {"WOR": 10.0, "M": 0.1,  "V": 0.3, "Y": 337.73,  "C_Ref3": 0.99, "C_Eq16": 0.98},
    {"WOR": 10.0, "M": 0.1,  "V": 0.5, "Y": 136.68,  "C_Ref3": 0.96, "C_Eq16": 0.95},
    {"WOR": 10.0, "M": 0.1,  "V": 0.8, "Y": 21.53,   "C_Ref3": 0.77, "C_Eq16": 0.81},
    {"WOR": 10.0, "M": 1.0,  "V": 0.3, "Y": 177.26,  "C_Ref3": 0.97, "C_Eq16": 0.96},
    {"WOR": 10.0, "M": 1.0,  "V": 0.5, "Y": 65.67,   "C_Ref3": 0.92, "C_Eq16": 0.92},
    {"WOR": 10.0, "M": 1.0,  "V": 0.8, "Y": 8.52,    "C_Ref3": 0.63, "C_Eq16": 0.65},
    {"WOR": 10.0, "M": 10.0, "V": 0.3, "Y": 30.82,   "C_Ref3": 0.86, "C_Eq16": 0.85},
    {"WOR": 10.0, "M": 10.0, "V": 0.5, "Y": 10.60,   "C_Ref3": 0.67, "C_Eq16": 0.69},
    {"WOR": 10.0, "M": 10.0, "V": 0.8, "Y": 1.21,    "C_Ref3": 0.23, "C_Eq16": 0.19},
    {"WOR": 50.0, "M": 0.1,  "V": 0.3, "Y": 1636.70, "C_Ref3": 0.99, "C_Eq16": 0.99},
    {"WOR": 50.0, "M": 0.1,  "V": 0.5, "Y": 662.39,  "C_Ref3": 0.99, "C_Eq16": 0.99},
    {"WOR": 50.0, "M": 0.1,  "V": 0.8, "Y": 104.33,  "C_Ref3": 0.90, "C_Eq16": 0.94},
    {"WOR": 50.0, "M": 1.0,  "V": 0.3, "Y": 859.04,  "C_Ref3": 0.99, "C_Eq16": 0.99},
    {"WOR": 50.0, "M": 1.0,  "V": 0.5, "Y": 318.25,  "C_Ref3": 0.98, "C_Eq16": 0.98},
    {"WOR": 50.0, "M": 1.0,  "V": 0.8, "Y": 41.29,   "C_Ref3": 0.85, "C_Eq16": 0.88},
    {"WOR": 50.0, "M": 10.0, "V": 0.3, "Y": 149.36,  "C_Ref3": 0.96, "C_Eq16": 0.96},
    {"WOR": 50.0, "M": 10.0, "V": 0.5, "Y": 51.37,   "C_Ref3": 0.89, "C_Eq16": 0.90},
    {"WOR": 50.0, "M": 10.0, "V": 0.8, "Y": 5.86,    "C_Ref3": 0.56, "C_Eq16": 0.56},
]
