'''
GHOST wrapper example for Kolmogorov Flow
'''

import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import SPECTER
import params as pm

# Initialize solver
grid = sp.Grid2D_semi(Lx=pm.Lx, Lz=pm.Lz, Nx=pm.Nx, Nz=pm.Nz, dt=pm.dt)
solver = SPECTER(grid,
                 nprocs=pm.nprocs,
                 ra=pm.ra,
                 pr=pm.pr,
                 gamma=1.,
                 solver='BOUSS',
                 ftypes=['vx', 'vz', 'th'],
                 precision='double',
                 ext=5)

# Generate initial condition with no slip boundaries
vx = np.zeros((grid.Nx, grid.Nz))
vz = np.zeros((grid.Nx, grid.Nz))
rng = np.random.default_rng(seed=42)

noise_amp = 1e-3
noise = noise_amp * rng.standard_normal((grid.Nx, grid.Nz))

# Enforce zero perturbation at top and bottom
X, Z = np.meshgrid(grid.xx, grid.zi, indexing='ij')
window = np.sin(np.pi * Z / grid.Lz)
theta = noise * window

fields_i = [vx, vz, theta]

# If you only need last fields
fields = solver.evolve(fields_i, pm.Tevolve)

# If you need intermediate fields
# fields = solver.evolve(fields_i, pm.Tevolve, opath=pm.opath, bpath=pm.bpath, ostep=pm.ostep, bstep=pm.bstep)

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
