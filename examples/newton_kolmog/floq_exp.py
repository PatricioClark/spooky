'''
Newton-Hookstep solver for Kolmogorov flow
==========================================
A variable-order RK scheme is used for time integration,
and the 2/3 rule is used for dealiasing.
'''


import numpy as np
import matplotlib.pyplot as plt
import os

import pySPEC as ps
from pySPEC.solvers import KolmogorovFlow
from pySPEC.methods import DynSys
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
solver = KolmogorovFlow(pm)
newt = DynSys(pm, solver)

# Load converged orbit

iN = 18 # Newton iteration of converged orbit
path = f'output/iN{iN:02}/'
fields = solver.load_fields(path, 0)
T, sx = newt.get_restart_values(iN) # Get period and shift from converged Newton iteration

U = newt.flatten(fields)
if pm.sx is not None: # If searching for RPOs
    X = np.append(U, [T, sx])
else:
    X = np.append(U, T) # If searching for UPOs with 0 shift

# Calculate n Floquet exponents
n = 50
eigval_H, eigvec_H, Q = newt.floq_exp(X, n, tol = 1e-10)

# Save Floquet exponents
spath = f'floq/iN{iN:02}/'
os.makedirs(spath, exist_ok=True)
np.save(f'{spath}floq_exp_{n}.npy', eigval_H)
np.save(f'{spath}eigvec_H_{n}.npy', eigvec_H)
np.save(f'{spath}Q_{n}.npy', Q)
