'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import GHOST
import params as pm

# Initialize solver
grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
solver = GHOST(grid, nu=pm.nu, nprocs=pm.nprocs, precision=pm.precision, ext=pm.ext)

# Initial velocity field
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)

fields = [uu, vv]
#Apply solenoidal projection
ffields = [grid.forward(field) for field in fields]
ffields = grid.inc_proj(ffields)
fields_i = [grid.inverse(ff) for ff in ffields]

# If you only need last fields
# fields = solver.evolve(fields_i, pm.Tevolve)

# If you need intermediate fields
fields = solver.evolve(fields, pm.Tevolve, opath=pm.opath, ostep=pm.ostep, bstep=pm.bstep)

# Plot initial and final fields
for field_i, field_f, ftype in zip(fields_i, fields, solver.ftypes):
    fig = plt.figure(figsize = (10, 5))

    plt.subplot(1,2,1)
    plt.imshow(field_i.T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(ftype)

    plt.subplot(1,2,2)
    plt.imshow(field_f.T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(ftype)
    plt.savefig(f'{ftype}.png', dpi = 300)
    plt.show()
