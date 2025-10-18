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
# Null initial velocity field
# uu = np.zeros_like(grid.xx)
# vv = np.zeros_like(grid.xx)


fields = [uu, vv]
#Apply solenoidal projection
ffields = [grid.forward(field) for field in fields]
ffields = grid.inc_proj(ffields)
fields_i = [grid.inverse(ff) for ff in ffields]

# If you only need last fields
# fields = solver.evolve(fields_i, pm.T)

# If you need intermediate fields
fields = solver.evolve(fields, pm.T, pm.ipath, pm.opath, pm.bstep, pm.ostep, pm.sstep, pm.bpath, pm.spath)

# Plot initial and final fields
for field_i, field, ftype in zip(fields_i, fields, solver.ftypes):
    fig = plt.figure(figsize = (12, 6))

    plt.subplot(1,3,1)
    plt.imshow(field_i.T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(ftype)

    plt.subplot(1,3,2)
    plt.imshow(field.T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(ftype)

    plt.subplot(1,3,3)
    plt.imshow((field_i-field).T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(ftype)
    plt.savefig(os.path.join(pm.opath,f'{ftype}.png'))
    plt.show()
