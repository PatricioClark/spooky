'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import spooky as ps
from spooky.solvers import GHOST
from spooky.methods import DynSys
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
grid = ps.Grid2D(pm)
solver = GHOST(pm, solver='HD', dimension=2)
newt = DynSys(pm, solver)

# Load initial conditions
if pm.restart_iN == 0:
    # Start Newton Solver from initial guess
    if pm.input_type == 'ps':
        fields = solver.load_fields(pm.input, pm.stat)
    elif pm.input_type == 'v':
        uu = np.load(os.path.join(pm.input, f'uu.{pm.stat:0{pm.ext}}.npy'))
        vv = np.load(os.path.join(pm.input, f'vv.{pm.stat:0{pm.ext}}.npy'))
        fields = [uu, vv]
    else:
        raise ValueError("Invalid input type. Use 'v' for velocity or 'ps' for streamfunction.")
    sx = pm.sx # Set initial guess for shift
    # Create directories
    newt.mkdirs()
    newt.write_header()

else:
    # Restart Newton Solver from last iteration at index 0 (start of evolution)
    restart_path = f'output/iN{pm.restart_iN:02}/'
    fields = solver.load_fields(restart_path, 0)
    _, sx = newt.get_restart_values(pm.restart_iN) # Get period and shift from last Newton iteration

U = newt.flatten(fields)
X = newt.form_X(U, sx=sx)

# Iterate Newton Solver
X = newt.run_newton(X)