"""
Newton-Hookstep solver for Kolmogorov flow
==========================================
A variable-order RK scheme is used for time integration,
and the 2/3 rule is used for dealiasing.
"""

import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np

from spooky.solvers import KolmogorovFlow
from spooky.methods import DynSys, Parameters

def load_config(path):
    """Safely load a YAML configuration file."""
    with open(path, "r") as f:
        dic = yaml.safe_load(f)
    return SimpleNamespace(**dic)

def main():
   #  Load configs 
    pm_solver = load_config("params_kolmog.yaml")   # Solver physics params
    # kolmog_cfg = SimpleNamespace(**kolmog_cfg)
    pm_solver.Lx *= 2 * np.pi
    pm_solver.Ly *= 2 * np.pi

    pm = load_config("params_newton.yaml")   # Newton-Krylov params

    #  Initialize solver and parameter containers 
    solver = KolmogorovFlow(pm_solver)
    # pm = Parameters(newton_cfg)

    #  Initialize the dynamical system wrapper 
    newt = DynSys(pm, solver)

    #  Load Initial Conditions
    if pm.restart_iN == 0:
        print("Starting new Newton-Krylov run", file=sys.stdout)
        fields = solver.load_fields(pm.input_dir, pm.start_idx)
        T, sx = pm.T, pm.sx
    else:
        print(f"Restarting from Newton iteration {pm.restart_iN}", file=sys.stdout)
        restart_path = os.path.join(pm.output_dir, f"iN{pm.restart_iN:02}")
        fields = solver.load_fields(restart_path, 0)
        T, sx = newt.get_restart_values(pm.restart_iN)

    #  Form Initial State Vector
    U = newt.flatten(fields)
    if pm.sx is not None:
        X = newt.form_X(U, T, sx)
    else:
        X = newt.form_X(U, T)

    #  Run Newton Solver
    print("Running Newton-Krylov solver", file=sys.stdout)
    newt.run_newton(X)
    print("Newton iteration completed successfully.", file=sys.stdout)

if __name__ == "__main__":
    main()
