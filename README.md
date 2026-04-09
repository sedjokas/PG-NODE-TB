# PG-NODE for Tuberculosis Transmission Dynamics

**Physics-Guided Neural Ordinary Differential Equations applied to TB epidemiology**

> Kasereka S.K. and K. Kyamakya  
> *"PG-NODE: Physics-Guided Neural Ordinary Differential Equations for Tuberculosis Transmission Dynamics"*  
> International Workshop on AI & Mathematical Methods for Real‑world Impact (AI2M4RI’2026)  
> Procedia Computer Science, August 2026, Athens, Greece.

---

## Overview

This repository contains the complete simulation code accompanying the paper above.
It implements a **Physics-Guided Neural ODE (PG-NODE)** framework applied to a classical
**SLIR** (Susceptible, Latent, Infectious, Recovered) compartmental TB model.

The key idea is **hybridization**: known epidemiological structure is preserved explicitly
in the ODE right-hand side, while unknown, time-varying, or unmodeled components are
represented by neural network functions embedded within the differential equation.

### Three Simulation Scenarios

| Scenario | Description | Key result |
|----------|-------------|------------|
| **1** | Classical SLIR vs. PG-NODE with learned time-varying transmission rate | R0 reduced from 3.61 to 2.24 |
| **2** | Neural correction for unmodeled treatment and relapse dynamics | 27% RMSE reduction vs. classical SLIR |
| **3** | 20-year intervention policy forecasting under 4 strategies | PG-NODE combined strategy averts 51k cases; R0 to 1.49 |

---

## Repository Structure

```
PG-NODE-TB/
├── run_all.py               Main entry point (run all figures)
├── requirements.txt         Python dependencies
├── README.md                This file
├── .gitignore
│
├── pgnode_tb/               Python package
│   ├── __init__.py
│   ├── parameters.py        Baseline epidemiological parameters
│   ├── models.py            ODE right-hand side functions
│   │                          - SLIR classical
│   │                          - SLIRT extended (ground truth)
│   │                          - PG-NODE Scenario 1 (learned beta)
│   │                          - PG-NODE Scenario 2 (neural correction)
│   │                          - PG-NODE Scenario 3 (optimal policy)
│   ├── analysis.py          R0, sensitivity indices, equilibria
│   ├── scenarios.py         Simulation scenarios 
│   └── utils.py             Plotting helpers, RMSE, figure saving
│
└── imgs/                    Output figures (PDF + PNG, gitignored)
    ├── scenario1_pgnode.{pdf,png}
    ├── scenario2_correction.{pdf,png}
    └── scenario3_forecast.{pdf,png}
```

---

## Installation

```bash
git clone https://github.com/sedjokas/PG-NODE-TB.git
cd PG-NODE-TB
pip install -r requirements.txt
```

**Dependencies** (all standard scientific Python):

| Package | Version |
|---------|---------|
| numpy | >= 1.24 |
| scipy | >= 1.11 |
| matplotlib | >= 3.7 |

No deep learning framework is required for this proof-of-concept.
The neural components are analytically prescribed functions; a follow-up
with full gradient-descent training via `torchdiffeq` is planned.

---

## Quick Start

```bash
# Run all scenarios and generate all figures
python run_all.py

# Specify output directory
python run_all.py --outdir results/

# Run a single scenario
python run_all.py --scenario 1   # Scenario 1 only
python run_all.py --scenario 2   # Scenario 2 only
python run_all.py --scenario 3   # Scenario 3 only

```

Expected output (terminal):

```
============================================================
  PG-NODE for TB — Simulation Runner
  Kasereka et al., Proof of Concept
============================================================

=======================================================
  SLIR TB MODEL — MATHEMATICAL ANALYSIS SUMMARY
=======================================================

  Basic Reproduction Number: R0 = 3.6142

  Disease-Free Equilibrium (DFE):
    S* = 666,667,  L* = I* = R* = 0

  Stability:
    DFE is unstable; endemic equilibrium exists (R0 > 1).

  Normalized Sensitivity Indices of R0:
    Upsilon_beta  = +1.0000  ####################
    Upsilon_k     = +0.1579  ###
    Upsilon_gamma = -0.8584  #################
    Upsilon_d     = -0.1288  ##
    Upsilon_mu    = -0.1707  ###
=======================================================

Running Scenario 1: time-varying beta_theta(t) ...
  Saved: scenario1_pgnode.pdf / scenario1_pgnode.png  ->  imgs/

Running Scenario 2: neural correction for SLIRT dynamics ...
  Saved: scenario2_correction.pdf / scenario2_correction.png  ->  imgs/

Running Scenario 3: intervention policy forecasting ...
  Saved: scenario3_forecast.pdf / scenario3_forecast.png  ->  imgs/

Done.  All figures saved to: imgs/
```

---

## Baseline Parameters

| Parameter | Symbol | Baseline | Unit | Source |
|-----------|--------|----------|------|--------|
| Recruitment rate | Lambda | 10,000 | yr^{-1} | Anderson & May 1991 |
| Natural mortality | mu | 0.015 | yr^{-1} | WHO 2025 |
| Transmission rate | beta | 5.0 | yr^{-1} | Blower et al. 1995 |
| Progression rate | k | 0.08 | yr^{-1} | Feng et al. 2000 |
| Recovery rate | gamma | 1.0 | yr^{-1} | Castillo-Chavez 2002 |
| TB-induced mortality | d | 0.15 | yr^{-1} | Dye & Williams 2000 |
| Treatment initiation | tau | 0.80 | yr^{-1} | Cohen & Murray 2004 |
| Relapse rate | delta | 0.03 | yr^{-1} | Castillo-Chavez 2002 |

Computed baseline: **R0 = 3.61**

---

## Mathematical Model

The SLIR ODE system:

```
dS/dt = Lambda - beta * S * I / N - mu * S
dL/dt = beta * S * I / N - (k + mu) * L
dI/dt = k * L - (gamma + mu + d) * I
dR/dt = gamma * I - mu * R
```

Basic reproduction number (Next-Generation Matrix method):

```
R0 = beta * k / [(k + mu) * (gamma + mu + d)]
```

PG-NODE extension:

```
dx/dt = f_mech(x; theta_known) + f_theta(t, x, u; theta_nn)
```

where `f_mech` is the SLIR right-hand side with fixed parameters and
`f_theta` is a neural network learning unknown/time-varying components
while preserving mass conservation: S + L + I + R = N.

---

## Key Results

**Scenario 1:** PG-NODE captures a 38% reduction in transmission following
a public-health intervention (year 8), bringing R0 from 3.61 to 2.24. The
classical SLIR converges to a fixed endemic equilibrium and misses this decline.

**Scenario 2:** The PG-NODE neural correction achieves RMSE = 5.20k against
the SLIRT ground truth, compared to RMSE = 7.10k for classical SLIR, a
**27% improvement** without adding explicit compartments.

**Scenario 3:** Over 20 years, the PG-NODE combined optimal strategy (D)
averts **51,000 TB cases** and reduces R0 to 1.49, outperforming contact
reduction alone (17,500 averted) and approaching the elimination threshold.

---

## Citation

If you use this code or the associated paper, please cite:

```bibtex
@inproceedings{kasereka2026pgnode,
  author    = {Kasereka, Selain K. and Kyamakya, Kyandoghere},
  title     = {{PG-NODE}: Physics-Guided Neural Ordinary Differential Equations
               for Tuberculosis Transmission Dynamics: A Proof of Concept},
  booktitle = {International Workshop on AI & Mathematical Methods for Real‑world Impact (AI2M4RI’2026)},
  series    = {Procedia Computer Science},
  year      = {2026},
  address   = {Athens, Greece},
  publisher = {Elsevier}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact Us: 

**Prof. Selain K. Kasereka**  
ABIL Research Center  
University of Kinshasa, DR Congo  
Email: selain.kasereka@unikin.ac.cd  
ABIL Research Center

**Prof. Kyandoghere Kyamakya**  
Institute of Smart Systems Technologies  
University of Klagenfurt, Austria  
Email: kyandoghere.kyamakya@aau.at  
ABIL Research Center
