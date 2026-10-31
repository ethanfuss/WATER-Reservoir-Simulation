# WATER: Predictive 3D Waterflood Simulator & Tracer Design Dashboard

A modular Python reservoir engineering simulation and interactive Streamlit dashboard based on **ARCO Resources Technology Technical Service Report TSR 84-28** (*"A Predictive Model for Calculation of Waterflood Performance Using a Handheld Calculator"*) by M. R. Fassihi and W. J. O'Brien, and **SPE 13945** (*"New Correlations for Calculation of Vertical Coverage and Areal Sweep Efficiency"*).

---

## Overview

Evaluating secondary oil recovery via waterflooding in multilayered reservoirs traditionally required either computationally expensive 3D numerical simulators or tedious manual graphical interpolations across Dykstra-Parsons and Dyes, Caudle, & Erickson nomographs.

This project modernizes the analytical **WATER** algorithm into a modular Python physics engine and interactive visual dashboard. It condenses empirical sweep charts into closed-form nonlinear regression correlations to forecast oil recovery, water-oil ratios ($WOR$), flood-front breakthrough timing, and radioactive tracer injection requirements in real time.

---

## Core Physics & Methodology

1. **Rock Heterogeneity & Relative Permeability (`core/rock_fluid.py`)**
   * **Permeability Variation ($V$):** Fits core sample permeabilities to a log-normal distribution (Subroutine `KVA`, Eq. 1) to compute the Dykstra-Parsons heterogeneity factor:
     $$V = \frac{k_{50} - k_{84.1}}{k_{50}} = \frac{10^{\overline{\log k}} - 10^{\overline{\log k} - \sigma}}{10^{\overline{\log k}}}$$
   * **Normalized Water Saturation ($S_{wn}$) & Rel-Perm Fit:** Normalizes saturation between irreducible water ($S_{wi}$) and residual oil ($S_{or}$) via $S_{wn} = \frac{S_w - S_{wi}}{1 - S_w - S_{or}}$ (Subroutine `SWN`) and fits laboratory relative permeability ratios via power-law regression (Subroutine `LIN1`, Eq. 2):
     $$\frac{k_{ro}}{k_{rw}} = a \cdot S_{wn}^{-b}$$

2. **Buckley-Leverett Frontal Advance (`core/frac_flow.py`)**
   * Computes the fractional flow curve ($f_w$, Eq. 3) and its analytical derivative ($f'_w = \partial f_w / \partial S_w$, Eq. 4).
   * Solves the Welge tangent construction (Eqs. 5–10) to pinpoint flood-front breakthrough saturation ($S_{wbt}$), fractional flow at breakthrough ($f_{wbt}$), average saturation behind the front ($\overline{S}_{wbt}$), and post-breakthrough saturation states.

3. **Closed-Form Sweep Efficiencies (`core/sweep_efficiency.py`)**
   * **Areal Sweep Efficiency ($E_a$):** Implements Fassihi's logarithmic correlation (Eqs. 14–15, Table 1) across **5-Spot**, **Direct Line Drive (DLD)**, and **Staggered Line Drive (SLD)** geometries for $0 < M \le 10$:
     $$\frac{1 - E_a}{E_a} = \left[a_1 \ln(M + a_2) + a_3\right] f_w + a_4 \ln(M + a_5) + a_6$$
   * **Vertical Sweep Efficiency / Coverage ($C$):** Evaluates the de Souza & Brigham grouping parameter ($Y$, Eq. 17) and inverts Fassihi's closed-form coverage correlation (Eqs. 16, 18–21) across four piecewise regimes for $0 < M \le 10$ and $0.3 \le V \le 0.8$:
     $$Y = \frac{(WOR + 0.4)(18.948 - 2.499V)}{(M + 1.137 - 0.8094V) \cdot 10^{f(V)}} = a_1 C^{a_2} (1 - C)^{a_3}$$

4. **Volumetric Material Balance & Time Forecaster (`core/forecaster.py`)**
   * Computes Original Oil in Place ($\text{OOIP}$) and gas fill-up time ($t_{fil}$), then iterates across user-defined producing $WOR$ steps to compute cumulative oil recovery ($N_p$, Eq. 22), surface water-oil ratio ($F_{wo}$, Eq. 23), trapezoidal-integrated cumulative water produced ($W_p$, Eq. 24), cumulative water injected ($W_i$, Eq. 25), project time ($t$, Eq. 26), and incremental oil production rate ($Q_o$, Eq. 27):
     $$N_p = \frac{\Delta S_w}{B_o} \cdot E_a \cdot C \cdot V_p$$

5. **Radioactive Tracer Injection Design (`core/tracer.py`)**
   * Models inter-well tracer slugs (**HTO**, **Co-60**, **Co-57**, **Ni-63**) from Tables 10 and 12 to calculate total tagged water volume ($W_{tag}$), radioactive decay fraction ($\zeta = e^{-\lambda t_{bt}}$), required isotope activity ($Q$ in pCi and Ci), injection concentration ($C$ in pCi/bbl), and regulatory safety margins against Maximum Permissible Concentration ($\%\text{MPC}$).

---

## Repository Structure

```text
waterflood_project/
│
├── core/                           # Pure physics & math engine (no UI code)
│   ├── __init__.py                 # Package exports
│   ├── presets.py                  # Example 1, Example 2, Table 1, Table 2 & Table 7 presets
│   ├── rock_fluid.py               # Permeability variation V, Swn, a & b regression
│   ├── frac_flow.py                # Buckley-Leverett fw, fw', Welge tangent solver
│   ├── sweep_efficiency.py         # Fassihi correlations for Areal Sweep (Ea) & Coverage (C)
│   ├── forecaster.py               # Material balance, Np, Wp, Wi, Qo, and time (t)
│   └── tracer.py                   # Radioactive tracer volume, activity Q, and %MPC
│
├── tests/                          # Unit tests validated against TSR 84-28 benchmarks
│   ├── test_example1.py            # Validates Example 1 (Five-Spot, V = 0.50, Table 5, 6, 7 & Fig. 7)
│   ├── test_example2.py            # Validates Example 2 (SLD, V = 0.624, Fig. 11 & Table 12 Tracers)
│   └── test_coverage_table2.py     # Validates all 36 (WOR, M, V) -> C pairs in Table 2
│
├── app.py                          # Interactive Streamlit visual dashboard
├── requirements.txt                # Project dependencies for local & Streamlit Cloud execution
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation