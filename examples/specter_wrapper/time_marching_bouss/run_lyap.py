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

# Calculate n Lyapunov exponents
T = 1.0
n = 5
nsteps = 50

lyap_exp, D_KY, _ = lyap.lyapunov_exponents(fields_i, T, n, nsteps, tol = 1e-10)