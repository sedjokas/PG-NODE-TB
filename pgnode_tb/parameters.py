"""
parameters.py
=============
Baseline epidemiological parameters for the SLIR TB model.

All rates are expressed per year (yr^{-1}).
Values are drawn from the systematic review by Kasereka et al. (2026)
and the foundational TB modeling literature.

References
----------
Blower et al. (1995)         Nature Medicine 1:815-821.
Feng et al.  (2000)          Theoretical Population Biology 57:235-247.
Castillo-Chavez & Song (2002) Math. Biosci. Eng. 1:361-404.
Dye & Williams (2000)        PNAS 97:8180-8185.
WHO (2023)                   Global Tuberculosis Report 2023.
"""

from dataclasses import dataclass, field


@dataclass
class BaselineParams:
    """
    Container for the baseline SLIR model parameters.

    Attributes
    ----------
    lam   : Recruitment (birth) rate into the susceptible class (yr^{-1}).
    mu    : Natural (non-TB) per-capita mortality rate (yr^{-1}).
    beta  : Effective TB transmission rate (yr^{-1}).
    k     : Progression rate from latency L to active disease I (yr^{-1}).
    gamma : Recovery rate from active TB (yr^{-1}).
    d     : TB-induced additional mortality rate (yr^{-1}).
    tau   : Treatment initiation rate (used in SLIRT; yr^{-1}).
    delta : Relapse rate from treated class back to I (used in SLIRT; yr^{-1}).
    """
    lam:   float = 10_000.0   # recruitment rate                   [Anderson 1991]
    mu:    float = 0.015      # natural mortality                   [WHO 2023]
    beta:  float = 5.0        # effective contact/transmission rate [Blower 1995]
    k:     float = 0.08       # progression rate (latent -> active) [Feng 2000]
    gamma: float = 1.0        # recovery rate                       [Castillo-Chavez 2002]
    d:     float = 0.15       # TB-induced mortality                [Dye 2000]
    tau:   float = 0.80       # treatment initiation rate           [Cohen 2004]
    delta: float = 0.03       # relapse rate                        [Castillo-Chavez 2002]


@dataclass
class InitialConditions:
    """
    Initial compartment sizes for a high-burden country scenario.
    Total population N = 1,000,000.
    """
    S: float = 800_000.0   # susceptible
    L: float = 180_000.0   # latently infected
    I: float =  18_000.0   # active TB (infectious)
    R: float =   2_000.0   # recovered

    @property
    def N(self) -> float:
        return self.S + self.L + self.I + self.R

    def as_list(self) -> list:
        return [self.S, self.L, self.I, self.R]

    def as_list_slirt(self) -> list:
        """SLIRT initial state: S, L, I, T=0, R."""
        return [self.S, self.L, self.I, 0.0, self.R]


# Singleton instances used throughout the package
PARAMS = BaselineParams()
IC     = InitialConditions()
