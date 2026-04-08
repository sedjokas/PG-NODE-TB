"""
analysis.py
===========
Mathematical analysis tools for the SLIR TB model:

  - Basic reproduction number R0 (Next-Generation Matrix method).
  - Normalized sensitivity indices of R0.
  - Disease-free equilibrium (DFE).
  - Endemic equilibrium approximation.

Reference
---------
van den Driessche, P. & Watmough, J. (2002).
  Reproduction numbers and sub-threshold endemic equilibria for
  compartmental models of disease transmission.
  Mathematical Biosciences 180(1-2): 29-48.
"""

import numpy as np
from .parameters import BaselineParams, PARAMS


def compute_R0(p: BaselineParams = PARAMS) -> float:
    """
    Compute the basic reproduction number for the SLIR model.

    Using the Next-Generation Matrix (NGM) method with infected
    compartments (L, I):

        F = [[0, beta],   V = [[k+mu,        0    ],
             [0,   0 ]]        [-k,    gamma+mu+d  ]]

    R0 = rho(F V^{-1}) = beta * k / [(k + mu) * (gamma + mu + d)]

    Parameters
    ----------
    p : BaselineParams instance

    Returns
    -------
    float : R0
    """
    return (p.beta * p.k) / ((p.k + p.mu) * (p.gamma + p.mu + p.d))


def sensitivity_indices(p: BaselineParams = PARAMS) -> dict:
    """
    Compute normalized sensitivity indices of R0 with respect to each
    parameter theta:

        Upsilon_theta = (dR0/d_theta) * (theta / R0)

    Parameters
    ----------
    p : BaselineParams instance

    Returns
    -------
    dict : {parameter_name: sensitivity_index}
    """
    return {
        "beta":  +1.0,
        "k":     p.mu / (p.k + p.mu),
        "gamma": -p.gamma / (p.gamma + p.mu + p.d),
        "d":     -p.d     / (p.gamma + p.mu + p.d),
        "mu":    -(p.mu / (p.k + p.mu) + p.mu / (p.gamma + p.mu + p.d)),
    }


def disease_free_equilibrium(p: BaselineParams = PARAMS) -> dict:
    """
    Return the disease-free equilibrium (DFE).

        E0 = (Lambda/mu, 0, 0, 0)

    Parameters
    ----------
    p : BaselineParams instance

    Returns
    -------
    dict : {compartment: value}
    """
    return {"S": p.lam / p.mu, "L": 0.0, "I": 0.0, "R": 0.0}


def endemic_equilibrium_R0(beta_val: float, p: BaselineParams = PARAMS) -> float:
    """
    Compute R0 for an arbitrary beta value (useful for scenario comparisons).

    Parameters
    ----------
    beta_val : transmission rate to use
    p        : BaselineParams instance

    Returns
    -------
    float : R0 at the given beta
    """
    return (beta_val * p.k) / ((p.k + p.mu) * (p.gamma + p.mu + p.d))


def print_analysis_summary(p: BaselineParams = PARAMS) -> None:
    """
    Print a concise summary of the mathematical analysis results.
    """
    R0  = compute_R0(p)
    dfe = disease_free_equilibrium(p)
    si  = sensitivity_indices(p)

    print("=" * 55)
    print("  SLIR TB MODEL — MATHEMATICAL ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"\n  Basic Reproduction Number: R0 = {R0:.4f}")
    print(f"\n  Disease-Free Equilibrium (DFE):")
    print(f"    S* = {dfe['S']:,.0f},  L* = I* = R* = 0")
    print(f"\n  Stability:")
    if R0 < 1:
        print("    DFE is locally asymptotically stable (R0 < 1).")
    else:
        print("    DFE is unstable; endemic equilibrium exists (R0 > 1).")
    print(f"\n  Normalized Sensitivity Indices of R0:")
    for name, val in si.items():
        bar = "#" * int(abs(val) * 20)
        sign = "+" if val >= 0 else ""
        print(f"    Upsilon_{name:<5} = {sign}{val:+.4f}  {bar}")
    print("=" * 55)
