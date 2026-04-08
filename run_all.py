#!/usr/bin/env python3
"""
run_all.py
==========
Main entry point for the PG-NODE TB proof-of-concept paper.

Usage
-----
    python run_all.py                    # use default output dir (imgs/)
    python run_all.py --outdir results   # custom output directory
    python run_all.py --scenario 1       # run a single scenario (1, 2, or 3)

Outputs
-------
All figures are saved as PDF and PNG in the output directory:
    scenario1_pgnode.{pdf,png}       — Scenario 1
    scenario2_correction.{pdf,png}   — Scenario 2
    scenario3_forecast.{pdf,png}     — Scenario 3
    pgnode_architecture.{pdf,png}    — Architecture diagram

Author
------
Selain K. Kasereka <selain.kasereka@aau.at>
Institute of Smart Systems Technologies, University of Klagenfurt
"""

import argparse
import sys
import time

from pgnode_tb.parameters import PARAMS, InitialConditions
from pgnode_tb.analysis   import print_analysis_summary
from pgnode_tb.scenarios  import (
    run_scenario1,
    run_scenario2,
    run_scenario3,
    plot_architecture,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PG-NODE TB: generate all simulation figures.'
    )
    parser.add_argument(
        '--outdir', type=str, default='imgs',
        help='Output directory for figures (default: imgs/)'
    )
    parser.add_argument(
        '--scenario', type=int, choices=[0, 1, 2, 3], default=0,
        help='Run a single scenario (0 = all, 1/2/3 = specific scenario)'
    )
    parser.add_argument(
        '--no-arch', action='store_true',
        help='Skip the architecture diagram'
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    p     = PARAMS
    ic    = InitialConditions()
    start = time.time()

    print("\n" + "=" * 60)
    print("  PG-NODE for TB — Simulation Runner")
    print("  Kasereka et al., MobiSPC 2026")
    print("=" * 60 + "\n")

    # Print mathematical analysis summary
    print_analysis_summary(p)
    print()

    run_s1 = args.scenario in (0, 1)
    run_s2 = args.scenario in (0, 2)
    run_s3 = args.scenario in (0, 3)
    run_ar = (args.scenario == 0) and (not args.no_arch)

    endemic_state = None   # will be populated by scenario 1 if run

    # ---- Scenario 1 ----
    if run_s1:
        print("Running Scenario 1: time-varying beta_theta(t) ...")
        res1 = run_scenario1(p=p, ic=ic, outdir=args.outdir)
        # Extract endemic state from end of classical SLIR for scenario 3
        sol_cls = res1['sol_classical_for_s3']
        endemic_state = [sol_cls.y[i][-1] for i in range(4)]
        print()

    # ---- Scenario 2 ----
    if run_s2:
        print("Running Scenario 2: neural correction for SLIRT dynamics ...")
        run_scenario2(p=p, ic=ic, outdir=args.outdir)
        print()

    # ---- Scenario 3 ----
    if run_s3:
        print("Running Scenario 3: intervention policy forecasting ...")
        run_scenario3(p=p, x0_endemic=endemic_state, outdir=args.outdir)
        print()

    # ---- Architecture diagram ----
    if run_ar:
        print("Generating PG-NODE architecture diagram ...")
        plot_architecture(outdir=args.outdir)
        print()

    elapsed = time.time() - start
    print(f"Done.  All figures saved to: {args.outdir}/")
    print(f"Total elapsed time: {elapsed:.1f}s\n")


if __name__ == '__main__':
    main()
