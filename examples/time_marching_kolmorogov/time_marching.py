'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.solvers import KolmogorovFlow

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))
pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L
pm.nu = 1.0/40.0

# Initialize solver
grid   = ps.Grid2D(pm)
solver = KolmogorovFlow(pm)

# Initial conditions
uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
fu = grid.forward(uu)
fv = grid.forward(vv)
fu, fv = grid.inc_proj([fu, fv])
uu = grid.inverse(fu)
vv = grid.inverse(fv)
fields = [uu, vv]

# Plot initial conditions
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(uu.T, cmap='viridis')
ax[0].set_title('uu')
ax[1].imshow(vv.T, cmap='viridis')
ax[1].set_title('vv')
plt.savefig(f'fields_ic.png', dpi = 300)
plt.show()


# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep)

# # Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.figure()
plt.plot(bal[0], bal[1])
plt.savefig('balance.png', dpi = 300)

# Plot fields
uu, vv = fields
u2 = uu**2 + vv**2
plt.figure()
plt.imshow(u2)
plt.savefig('u2.png', dpi = 300)
plt.show()

# Plot output
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(uu, cmap='viridis')
ax[0].set_title('uu')
ax[1].imshow(vv, cmap='viridis')
ax[1].set_title('vv')
plt.savefig(f'fields.png', dpi = 300)
plt.show()

oz = solver.oz([uu, vv])

plt.figure()
plt.imshow(oz, cmap='viridis')
plt.title('oz')
plt.savefig(f'oz.png', dpi = 300)
plt.show()
