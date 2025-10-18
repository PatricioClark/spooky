'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import pySPEC as ps
from pySPEC.solvers import GHOST
from pySPEC.methods import DynSys
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
grid = ps.Grid2D(pm)
solver = GHOST(pm)
lyap = DynSys(pm, solver)

# If loading initial velocity field
# path = 'input'
# fields = solver.load_fields(path, 0)

# If writing initial velocity field
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)

fields = [uu, vv]
#Apply solenoidal projection
ffields = [grid.forward(field) for field in fields]
ffields = grid.inc_proj(ffields)
fields_i = [grid.inverse(ff) for ff in ffields]

# Calculate n Lyapunov exponents
n = 5
T = 1.0

eigval_H, eigvec_H, Q = lyap.lyap_exp(fields, T, n, tol = 1e-10)

# Save Lyapunov exponents
spath = f'lyap/'
os.makedirs(spath, exist_ok=True)
np.save(f'{spath}lyap_exp.npy', eigval_H)
np.save(f'{spath}eigvec_H.npy', eigvec_H)
np.save(f'{spath}Q.npy', Q)
