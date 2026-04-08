"""
pgnode_tb — Physics-Guided Neural ODE for TB transmission modeling.

Proof-of-concept accompanying the paper:
  "PG-NODE: Physics-Guided Neural Ordinary Differential Equations
   for Tuberculosis Transmission Dynamics: A Proof of Concept"
  Kasereka et al., MobiSPC 2026, Procedia Computer Science.

Modules
-------
parameters  : Baseline biological/epidemiological parameters.
models      : ODE right-hand side functions (SLIR, SLIRT, PG-NODE variants).
analysis    : R0 computation and sensitivity indices.
scenarios   : Three simulation scenarios and architecture diagram.
utils       : Plotting helpers.
"""

__version__ = "1.0.0"
__author__  = "Selain K. Kasereka"
__email__   = "selain.kasereka@aau.at"
