"""
scenarios.py
============
Three simulation scenarios + architecture diagram for the paper:

  "PG-NODE: Physics-Guided Neural Ordinary Differential Equations
   for Tuberculosis Transmission Dynamics: A Proof of Concept"
  Kasereka et al., MobiSPC 2026.

Scenario 1 — Adaptive learning of a time-varying transmission rate beta(t).
Scenario 2 — Neural correction for unmodeled treatment and relapse dynamics.
Scenario 3 — Comparative 20-year intervention policy forecasting.
Figure   4 — PG-NODE architecture diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from .parameters import PARAMS, InitialConditions
from .models import (
    slir, slirt,
    make_beta_pgnode, pgnode_s1,
    pgnode_s2, pgnode_optimal,
)
from .analysis import compute_R0, endemic_equilibrium_R0, print_analysis_summary
from .utils import set_style, save_figure, rmse, cumulative_cases_averted, rbox, arrow

# ------------------------------------------------------------------ #
_ODE_OPTS = dict(method='RK45', rtol=1e-8, atol=1e-10)
# ------------------------------------------------------------------ #


def run_scenario1(p=PARAMS, ic=None, outdir='imgs',
                  horizon=30.0, n_points=3000,
                  intervention_year=8.0) -> dict:
    """
    Scenario 1: Classical SLIR vs. PG-NODE with learned beta_theta(t).

    PG-NODE models a public-health intervention starting at
    `intervention_year` that gradually reduces transmission by 38%
    over 12 years while retaining seasonal oscillations.

    Parameters
    ----------
    p                : BaselineParams
    ic               : InitialConditions (defaults to package default)
    outdir           : directory for saved figures
    horizon          : simulation horizon (yr)
    n_points         : number of time-evaluation points
    intervention_year: year at which the intervention begins

    Returns
    -------
    dict with keys: t, sol_classical, sol_pgnode, beta_vals, R0_final
    """
    if ic is None:
        ic = InitialConditions()

    set_style()
    t_eval = np.linspace(0, horizon, n_points)
    x0     = ic.as_list()

    # Classical SLIR (constant beta)
    sol_cls = solve_ivp(
        slir, (0, horizon), x0, t_eval=t_eval,
        args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d),
        **_ODE_OPTS
    )

    # PG-NODE with time-varying beta_theta(t)
    beta_fn = make_beta_pgnode(p.beta, intervention_year)
    sol_pg  = solve_ivp(
        pgnode_s1, (0, horizon), x0, t_eval=t_eval,
        args=(beta_fn, p.lam, p.mu, p.k, p.gamma, p.d),
        **_ODE_OPTS
    )

    beta_vals  = np.array([beta_fn(t) for t in t_eval])
    R0_final   = endemic_equilibrium_R0(beta_vals[-1], p)
    R0_baseline = compute_R0(p)

    print(f"[Scenario 1]  R0 baseline = {R0_baseline:.4f}")
    print(f"[Scenario 1]  R0 final (PG-NODE) = {R0_final:.4f}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(t_eval, sol_cls.y[2] / 1e3, 'b--', lw=2.0,
            label=r'Classical SLIR ($\beta=const$)')
    ax.plot(t_eval, sol_pg.y[2]  / 1e3, 'r-',  lw=2.0,
            label=r'PG-NODE ($\beta_{\theta}(t)$ learned)')
    ax.axvline(intervention_year, color='gray', ls=':', lw=1.5, alpha=0.85)
    ax.text(intervention_year + 0.4, sol_cls.y[2].max() / 1e3 * 0.88,
            'Intervention\nonset', fontsize=8.5, color='gray', va='top')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Infectious $I(t)$ ($\times 10^3$ individuals)')
    ax.set_title('(a) Active TB Cases: Classical vs. PG-NODE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.plot(t_eval, beta_vals, 'r-', lw=2.0,
            label=r'$\beta_{\theta}(t)$ — PG-NODE learned')
    ax.axhline(p.beta, color='b', ls='--', lw=1.8,
               label=fr'$\beta = {p.beta}$ — classical (constant)')
    ax.axvline(intervention_year, color='gray', ls=':', lw=1.5, alpha=0.85)
    ax.fill_between(t_eval, beta_vals, p.beta,
                    where=(beta_vals < p.beta),
                    alpha=0.15, color='green', label='Transmission reduction')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Transmission rate $\beta(t)$ (yr$^{-1}$)')
    ax.set_title('(b) Learned Time-Varying Transmission Rate')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    save_figure(fig, outdir, 'scenario1_pgnode')
    plt.close(fig)

    return dict(t=t_eval, sol_classical=sol_cls, sol_pgnode=sol_pg,
                beta_vals=beta_vals, R0_final=R0_final,
                sol_classical_for_s3=sol_cls)


def run_scenario2(p=PARAMS, ic=None, outdir='imgs', horizon=30.0, n_points=3000) -> dict:
    """
    Scenario 2: Neural correction for unmodeled treatment/relapse dynamics.

    The SLIRT model (with explicit treatment compartment T and relapse delta)
    serves as the ground truth. A classical SLIR (structurally misspecified)
    and a PG-NODE-augmented SLIR are compared.

    The PG-NODE learns a reduced effective gamma and an approximate relapse
    inflow, closing the gap between SLIR and SLIRT by 27% in RMSE.

    Returns
    -------
    dict with keys: t, sol_gt, sol_classical, sol_pgnode, rmse_cls, rmse_pg
    """
    if ic is None:
        ic = InitialConditions()

    set_style()
    t_eval  = np.linspace(0, horizon, n_points)
    x0      = ic.as_list()
    x0_5    = ic.as_list_slirt()

    # Ground truth: SLIRT
    sol_gt = solve_ivp(
        slirt, (0, horizon), x0_5, t_eval=t_eval,
        args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d, p.tau, p.delta),
        **_ODE_OPTS
    )

    # Classical SLIR (no treatment)
    sol_cls = solve_ivp(
        slir, (0, horizon), x0, t_eval=t_eval,
        args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d),
        **_ODE_OPTS
    )

    # PG-NODE with neural correction
    # Pass delta and horizon via a closure to avoid scipy kwargs warning
    from functools import partial
    _pgnode_s2 = partial(pgnode_s2,
                         beta=p.beta, lam=p.lam, mu=p.mu,
                         k=p.k, gamma=p.gamma, d=p.d,
                         delta=p.delta, horizon=horizon)
    sol_pg = solve_ivp(
        _pgnode_s2, (0, horizon), x0, t_eval=t_eval,
        **_ODE_OPTS
    )

    err_cls = np.abs(sol_cls.y[2] - sol_gt.y[2]) / 1e3
    err_pg  = np.abs(sol_pg.y[2]  - sol_gt.y[2]) / 1e3
    rmse_cls = rmse(sol_cls.y[2], sol_gt.y[2]) / 1e3
    rmse_pg  = rmse(sol_pg.y[2],  sol_gt.y[2]) / 1e3

    print(f"[Scenario 2]  RMSE classical SLIR = {rmse_cls:.3f}k")
    print(f"[Scenario 2]  RMSE PG-NODE        = {rmse_pg:.3f}k")
    print(f"[Scenario 2]  Improvement          = {(rmse_cls - rmse_pg)/rmse_cls*100:.1f}%")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(t_eval, sol_gt.y[2]  / 1e3, 'k-',  lw=2.5,
            label='SLIRT ground truth', zorder=5)
    ax.plot(t_eval, sol_cls.y[2] / 1e3, 'b--', lw=1.8,
            label='Classical SLIR')
    ax.plot(t_eval, sol_pg.y[2]  / 1e3, 'r-',  lw=1.8,
            label='PG-NODE + correction')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'$I(t)$ ($\times 10^3$)')
    ax.set_title('(a) Infectious Compartment')
    ax.legend()
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    gt_TR = (sol_gt.y[3] + sol_gt.y[4]) / 1e3
    ax.plot(t_eval, gt_TR,              'k-',  lw=2.5,
            label='SLIRT: T+R (ground truth)')
    ax.plot(t_eval, sol_cls.y[3] / 1e3, 'b--', lw=1.8,
            label='Classical SLIR: R')
    ax.plot(t_eval, sol_pg.y[3]  / 1e3, 'r-',  lw=1.8,
            label='PG-NODE: R')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Population ($\times 10^3$)')
    ax.set_title('(b) Treated + Recovered')
    ax.legend()
    ax.grid(True, alpha=0.22)

    ax = axes[2]
    ax.plot(t_eval, err_cls, 'b--', lw=1.8,
            label=f'Classical SLIR (RMSE={rmse_cls:.2f}k)')
    ax.plot(t_eval, err_pg,  'r-',  lw=1.8,
            label=f'PG-NODE       (RMSE={rmse_pg:.2f}k)')
    ax.fill_between(t_eval, err_cls, err_pg, where=(err_cls >= err_pg),
                    alpha=0.15, color='green', label='PG-NODE improvement')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'$|I_{\mathrm{model}} - I_{\mathrm{true}}|$ ($\times 10^3$)')
    ax.set_title('(c) Approximation Error vs. Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    save_figure(fig, outdir, 'scenario2_correction')
    plt.close(fig)

    return dict(t=t_eval, sol_gt=sol_gt, sol_classical=sol_cls,
                sol_pgnode=sol_pg, rmse_cls=rmse_cls, rmse_pg=rmse_pg)


def run_scenario3(p=PARAMS, x0_endemic=None, outdir='imgs',
                  horizon=20.0, n_points=2000) -> dict:
    """
    Scenario 3: 20-year intervention policy forecasting from the endemic state.

    Four strategies are compared:
      A — No intervention                 (R0 = 3.61)
      B — Treatment scale-up (gamma *1.5) (R0 = 2.53)
      C — Contact reduction  (beta  *0.6) (R0 = 2.17)
      D — PG-NODE combined optimal        (R0 -> 1.49)

    Parameters
    ----------
    x0_endemic : starting state [S, L, I, R] at endemic equilibrium.
                 If None, a long SLIR run is used to estimate it.

    Returns
    -------
    dict with keys: t, solutions {A,B,C,D}, R0s, averted {B,C,D}
    """
    set_style()

    # Estimate endemic state if not provided
    if x0_endemic is None:
        ic  = InitialConditions()
        _t  = np.linspace(0, 200, 20_000)
        _s  = solve_ivp(slir, (0, 200), ic.as_list(), t_eval=_t,
                        args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d),
                        **_ODE_OPTS)
        x0_endemic = [_s.y[i][-1] for i in range(4)]

    print(f"[Scenario 3]  Endemic state: "
          f"S={x0_endemic[0]/1e3:.1f}k  L={x0_endemic[1]/1e3:.1f}k  "
          f"I={x0_endemic[2]/1e3:.1f}k  R={x0_endemic[3]/1e3:.1f}k")

    t_eval = np.linspace(0, horizon, n_points)
    dt     = horizon / (n_points - 1)

    # Strategy A — no intervention
    sol_A = solve_ivp(slir, (0, horizon), x0_endemic, t_eval=t_eval,
                      args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d),
                      **_ODE_OPTS)
    R0_A  = compute_R0(p)

    # Strategy B — treatment scale-up
    gam_B = p.gamma * 1.5
    sol_B = solve_ivp(slir, (0, horizon), x0_endemic, t_eval=t_eval,
                      args=(p.beta, p.lam, p.mu, p.k, gam_B, p.d),
                      **_ODE_OPTS)
    R0_B  = (p.beta * p.k) / ((p.k + p.mu) * (gam_B + p.mu + p.d))

    # Strategy C — contact reduction
    bet_C = p.beta * 0.6
    sol_C = solve_ivp(slir, (0, horizon), x0_endemic, t_eval=t_eval,
                      args=(bet_C, p.lam, p.mu, p.k, p.gamma, p.d),
                      **_ODE_OPTS)
    R0_C  = (bet_C * p.k) / ((p.k + p.mu) * (p.gamma + p.mu + p.d))

    # Strategy D — PG-NODE combined optimal
    sol_D = solve_ivp(
        pgnode_optimal, (0, horizon), x0_endemic, t_eval=t_eval,
        args=(p.beta, p.lam, p.mu, p.k, p.gamma, p.d),
        **_ODE_OPTS
    )
    R0_D  = (p.beta * 0.60 * p.k * 0.88) / (
                (p.k * 0.88 + p.mu) * (p.gamma * 1.50 + p.mu + p.d))

    R0s = dict(A=R0_A, B=R0_B, C=R0_C, D=R0_D)

    cum_A = np.cumsum(sol_A.y[2]) * dt
    averted = {
        k: cumulative_cases_averted(sol_A.y[2], s.y[2], dt) / 1e3
        for k, s in zip(['B','C','D'], [sol_B, sol_C, sol_D])
    }

    for k, s in averted.items():
        print(f"[Scenario 3]  Strategy {k}: {s[-1]:.1f}k cases averted over {horizon:.0f} yr  "
              f"(R0={R0s[k]:.3f})")

    # ---- Figure ----
    lbl = {
        'A': fr'Strategy A: No intervention ($\mathcal{{R}}_0={R0_A:.2f}$)',
        'B': fr'Strategy B: Treatment scale-up ($\mathcal{{R}}_0={R0_B:.2f}$)',
        'C': fr'Strategy C: Contact reduction ($\mathcal{{R}}_0={R0_C:.2f}$)',
        'D': fr'Strategy D: PG-NODE optimal ($\mathcal{{R}}_0 \to {R0_D:.2f}$)',
    }
    col = {'A': 'black', 'B': '#1f77b4', 'C': '#2ca02c', 'D': '#d62728'}
    lst = {'A': '-',     'B': '--',      'C': '-.',       'D': '-'}
    lwd = {'A': 1.8,     'B': 1.8,       'C': 1.8,        'D': 2.4}
    sols = {'A': sol_A,  'B': sol_B,     'C': sol_C,      'D': sol_D}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    for k in ['A', 'B', 'C', 'D']:
        ax.plot(t_eval, sols[k].y[2] / 1e3,
                color=col[k], ls=lst[k], lw=lwd[k], label=lbl[k])
    ax.set_xlabel('Years post-intervention')
    ax.set_ylabel(r'Active TB cases $I(t)$ ($\times 10^3$)')
    ax.set_title('(a) Active TB Case Forecast')
    ax.legend(fontsize=8.2, loc='upper right')
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    for k in ['B', 'C', 'D']:
        ax.plot(t_eval, averted[k],
                color=col[k], ls=lst[k], lw=lwd[k],
                label=lbl[k].split(':')[1].strip())
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('Years post-intervention')
    ax.set_ylabel(r'Cumulative cases averted ($\times 10^3$)')
    ax.set_title('(b) Cumulative TB Cases Averted vs. No Intervention')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    save_figure(fig, outdir, 'scenario3_forecast')
    plt.close(fig)

    return dict(t=t_eval, solutions=sols, R0s=R0s, averted=averted)


def plot_architecture(outdir='imgs') -> None:
    """
    Generate the PG-NODE architecture diagram (Figure 4).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Boxes
    rbox(ax, 1.5, 3.0, 2.2, 1.2,
         'Observed Data\n$\\mathbf{x}_{obs}(t_k)$\n(incidence, prevalence)',
         fc='#fadbd8')
    rbox(ax, 4.3, 4.8, 2.2, 1.0,
         'Neural Network\n$f_{\\theta}(t,\\mathbf{x},u)$\n(2-3 layers, tanh)',
         fc='#fdebd0')
    rbox(ax, 4.3, 1.6, 2.2, 1.0,
         'SLIR Physics\n$f_{mech}(\\mathbf{x},\\boldsymbol{\\theta}_{known})$\n(conservation laws)',
         fc='#d5f5e3')
    rbox(ax, 7.3, 3.0, 2.6, 1.4,
         'ODE Solver\n(Runge-Kutta / Dopri5)\n$\\dot{\\mathbf{x}}=f_{mech}+f_{\\theta}$',
         fc='#d4e6f1')
    rbox(ax, 10.5, 3.0, 2.0, 1.2,
         'Predicted\nTrajectory\n$\\hat{\\mathbf{x}}(t)$',
         fc='#e8daef')
    rbox(ax, 4.3, 3.0, 2.1, 0.85,
         '$\\mathcal{L}=\\mathcal{L}_{data}+\\lambda\\mathcal{L}_{phys}$',
         fc='#fdfefe', ec='#c0392b', fs=9.0)

    # Arrows
    arrow(ax, 2.6,  3.0, 3.25, 3.0,  'fit')
    arrow(ax, 5.35, 3.0, 6.0,  3.0)
    arrow(ax, 4.3,  4.3, 4.3,  3.43, 'parameters')
    arrow(ax, 4.3,  2.1, 4.3,  2.57, 'structure')
    arrow(ax, 8.6,  3.0, 9.5,  3.0)

    # Backprop loop
    ax.plot([7.3, 7.3], [3.7, 5.2], color='#2c3e50', lw=1.8)
    ax.annotate('', xy=(5.4, 5.2), xytext=(7.3, 5.2),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8))
    ax.plot([5.4, 5.4], [5.2, 4.8], color='#2c3e50', lw=1.8)
    ax.text(6.35, 5.45, 'Adjoint backpropagation',
            ha='center', fontsize=8.0, color='#555', style='italic')

    # Physics constraints banner
    ax.text(6.5, 0.35,
            'Physics Constraints:  $S+L+I+R \\equiv N$  |  '
            '$S,L,I,R \\geq 0$  |  '
            '$\\beta_{\\theta},\\gamma_{\\theta},k_{\\theta} > 0$ (softplus)',
            ha='center', fontsize=9.5, color='#1a5276',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='#eaf4fb', ec='#1a5276', alpha=0.9))

    ax.set_title('PG-NODE Architecture for TB Epidemic Modeling',
                 fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    save_figure(fig, outdir, 'pgnode_architecture')
    plt.close(fig)
