"""
models.py
=========
ODE right-hand side functions for:

  1. Classical SLIR model (Susceptible, Latent, Infectious, Recovered).
  2. Extended SLIRT model (adds Treatment compartment T with relapse).
  3. PG-NODE Scenario 1: learned time-varying transmission beta_theta(t).
  4. PG-NODE Scenario 2: neural correction for unmodeled treatment/relapse.
  5. PG-NODE Scenario 3: optimal combined intervention policy.

All functions follow the scipy.integrate.solve_ivp convention:
    f(t, y, *args) -> list[float]

Mass conservation
-----------------
For SLIR:  dN/dt = Lambda - mu*N - d*I  (N decreases due to TB mortality)
For SLIRT: same form with tau replacing gamma in the I equation.
"""

import numpy as np
from .parameters import BaselineParams, PARAMS


# ------------------------------------------------------------------ #
#  1. Classical SLIR
# ------------------------------------------------------------------ #
def slir(t, y, beta, lam, mu, k, gamma, d):
    """
    Classical SLIR TB model with demography.

    Parameters
    ----------
    t                : current time (yr)
    y                : state vector [S, L, I, R]
    beta, lam, mu,
    k, gamma, d      : model parameters (see BaselineParams)

    Returns
    -------
    list[float] : [dS/dt, dL/dt, dI/dt, dR/dt]
    """
    S, L, I, R = y
    N   = max(S + L + I + R, 1.0)
    foi = beta * S * I / N          # force of infection
    return [
        lam - foi - mu * S,
        foi - (k + mu) * L,
        k * L - (gamma + mu + d) * I,
        gamma * I - mu * R,
    ]


# ------------------------------------------------------------------ #
#  2. Extended SLIRT (ground truth for Scenario 2)
# ------------------------------------------------------------------ #
def slirt(t, y, beta, lam, mu, k, gamma, d, tau, delta):
    """
    Extended SLIRT model: adds Treatment compartment T.

    Compartments: S, L, I, T (under treatment), R (recovered).

    Parameters
    ----------
    tau   : treatment initiation rate (yr^{-1})
    delta : relapse rate from T back to I (yr^{-1})
    """
    S, L, I, T, Rc = y
    N   = max(S + L + I + T + Rc, 1.0)
    foi = beta * S * I / N
    return [
        lam - foi - mu * S,
        foi - (k + mu) * L,
        k * L + delta * T - (tau + mu + d) * I,
        tau * I - (gamma + delta + mu) * T,
        gamma * T - mu * Rc,
    ]


# ------------------------------------------------------------------ #
#  3. PG-NODE Scenario 1 — time-varying beta_theta(t)
# ------------------------------------------------------------------ #
def make_beta_pgnode(beta0, intervention_year, seasonal_amp=0.12,
                     reduction=0.38, ramp_years=12.0):
    """
    Factory: returns a callable beta_theta(t) representing the
    PG-NODE-learned time-varying transmission rate.

    Before the intervention: beta(t) = beta0 * (1 + seasonal_amp * sin(2pi*t))
    After  the intervention: transmission reduces gradually by `reduction`
                             over `ramp_years`, with damped seasonality.

    Parameters
    ----------
    beta0            : baseline transmission rate (yr^{-1})
    intervention_year: year at which the intervention begins
    seasonal_amp     : amplitude of seasonal oscillation (fraction)
    reduction        : maximum fractional reduction in beta post-intervention
    ramp_years       : time (yr) over which the reduction ramps to its maximum

    Returns
    -------
    callable  beta_theta(t) -> float
    """
    def beta_theta(t):
        seasonal = seasonal_amp * np.sin(2.0 * np.pi * t)
        if t < intervention_year:
            return max(0.1, beta0 * (1.0 + seasonal))
        else:
            progress = min((t - intervention_year) / ramp_years, 1.0)
            return max(0.5, beta0 * (1.0 - reduction * progress) * (1.0 + seasonal * 0.6))

    return beta_theta


def pgnode_s1(t, y, beta_theta, lam, mu, k, gamma, d):
    """
    PG-NODE SLIR with learned time-varying transmission beta_theta(t).

    Parameters
    ----------
    beta_theta : callable, returns beta value at time t
    """
    S, L, I, R = y
    N = max(S + L + I + R, 1.0)
    b = beta_theta(t)
    return [
        lam - b * S * I / N - mu * S,
        b * S * I / N - (k + mu) * L,
        k * L - (gamma + mu + d) * I,
        gamma * I - mu * R,
    ]


# ------------------------------------------------------------------ #
#  4. PG-NODE Scenario 2 — neural correction for unmodeled dynamics
# ------------------------------------------------------------------ #
def pgnode_s2(t, y, beta=5.0, lam=10000.0, mu=0.015, k=0.08, gamma=1.0, d=0.15,
              gam_correction=0.18, gam_ramp=8.0,
              relapse_factor=0.55, delta=0.03,
              horizon=30.0):
    """
    PG-NODE SLIR augmented with a neural correction term that approximates
    the effect of the unmodeled treatment compartment and relapse dynamics.

    The correction learns:
      - A reduced effective recovery rate:
            gamma_eff(t) = gamma - gam_correction * (1 - exp(-t/horizon * gam_ramp))
        which approximates the lower removal rate tau < gamma in SLIRT.
      - An approximate relapse inflow from R back to I:
            relapse_in(t) = delta * relapse_factor * R(t)

    Parameters
    ----------
    gam_correction : maximum reduction applied to gamma (learned)
    gam_ramp       : exponential ramp rate for the correction
    relapse_factor : scaling factor for the relapse inflow proxy
    delta          : relapse rate coefficient
    horizon        : simulation horizon (yr), used to normalize time
    """
    S, L, I, R = y
    N         = max(S + L + I + R, 1.0)
    tnorm     = min(t / horizon, 1.0)
    gamma_eff = gamma - gam_correction * (1.0 - np.exp(-tnorm * gam_ramp))
    relapse   = delta * relapse_factor * R
    foi       = beta * S * I / N
    return [
        lam - foi - mu * S,
        foi - (k + mu) * L,
        k * L + relapse - (gamma_eff + mu + d) * I,
        gamma_eff * I - mu * R - relapse,
    ]


# ------------------------------------------------------------------ #
#  5. PG-NODE Scenario 3 — optimal combined intervention
# ------------------------------------------------------------------ #
def pgnode_optimal(t, y, beta0, lam, mu, k0, gamma0, d,
                   beta_reduction=0.40, gamma_increase=0.50,
                   k_reduction=0.12, ramp_years=10.0):
    """
    PG-NODE optimal combined intervention strategy (Strategy D).

    Three parameters are co-optimized gradually over `ramp_years`:
      - beta(t)  : transmission reduced by beta_reduction  (e.g. 40%)
      - gamma(t) : recovery increased by gamma_increase    (e.g. 50%)
      - k(t)     : progression reduced by k_reduction      (e.g. 12%)

    This represents a coordinated policy package:
      contact-tracing + treatment scale-up + LTBI screening.

    Parameters
    ----------
    beta_reduction  : final fractional reduction in beta
    gamma_increase  : final fractional increase in gamma
    k_reduction     : final fractional reduction in k
    ramp_years      : duration (yr) of the ramp-up phase
    """
    S, L, I, R = y
    N    = max(S + L + I + R, 1.0)
    prog = min(t / ramp_years, 1.0)
    b_t  = max(0.5, beta0  * (1.0 - beta_reduction  * prog))
    g_t  = gamma0 * (1.0 + gamma_increase * prog)
    k_t  = k0     * (1.0 - k_reduction    * prog)
    return [
        lam - b_t * S * I / N - mu * S,
        b_t * S * I / N - (k_t + mu) * L,
        k_t * L - (g_t + mu + d) * I,
        g_t * I - mu * R,
    ]
