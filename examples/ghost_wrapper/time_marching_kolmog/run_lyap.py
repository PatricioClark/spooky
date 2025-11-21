'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import GHOST
from spooky.methods import DynSys
import params as pm

# Initialize solver
grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
solver = GHOST(grid, nu=pm.nu, nprocs=pm.nprocs, precision=pm.precision, ext=pm.ext)
lyap = DynSys(solver)

# Initial velocity field
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)

fields = [uu, vv]
#Apply solenoidal projection
ffields = [grid.forward(field) for field in fields]
ffields = grid.inc_proj(ffields)
fields_i = [grid.inverse(ff) for ff in ffields]

# Calculate n Lyapunov exponents
T = 1.0
n = 5
nsteps = 50

lyap_exp, D_KY = lyap.lyapunov_exponents(fields_i, T, n, nsteps, tol = 1e-10)