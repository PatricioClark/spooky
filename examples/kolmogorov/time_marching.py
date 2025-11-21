'''
2D Kolmgorov flow
'''

import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import KolmogorovFlow

import params as pm

grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
solver = KolmogorovFlow(grid, nu=pm.nu)

# Initial conditions
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
fu = grid.forward(uu)
fv = grid.forward(vv)
fu, fv = grid.inc_proj([fu, fv])
uu = grid.inverse(fu)
vv = grid.inverse(fv)
fields = [uu, vv]

# Evolve
fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep)

# Plot final fields
uu, vv = fields
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(uu.T, cmap='viridis')
ax[0].set_title('uu')
ax[1].imshow(vv.T, cmap='viridis')
ax[1].set_title('vv')
plt.savefig('fields.png', dpi = 300)
plt.show()
plt.close()
