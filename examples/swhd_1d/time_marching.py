'''
Pseudo-spectral solver for the 1D shallow water equations over topography
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import spooky as sp
from spooky.solvers import SWHD_1D

import params as pm

opath = 'outs'
os.makedirs(opath, exist_ok=True)

# Initialize solver
grid = sp.Grid1D(pm.Lx, pm.Nx, pm.dt)
xx = grid.xx

# Bottom topography: two Gaussian bumps
hb = (0.1*np.exp(-(xx - np.pi/1.4)**2/0.3**2) +
      0.05*np.exp(-(xx - np.pi/0.8)**2/0.2**2)
      )
solver = SWHD_1D(grid, g=pm.g, hb=hb, rkord=pm.rkord)

# Initial conditions: Gaussian pulse in velocity and free surface
uu = 2.5e-4*np.exp(-(xx - np.pi/2)**2/0.5**2)
hh = pm.h0 + 5e-5*np.exp(-(xx - np.pi/2)**2/0.5**2)
fields = [uu, hh]

# Evolve
fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep, ostep=pm.ostep,
                       bpath=opath, opath=opath)

# Plot final fields
uu, hh = fields
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(xx, hb)
ax[0].set_title('hb')
ax[1].plot(xx, uu)
ax[1].set_title('u')
ax[2].plot(xx, hh)
ax[2].set_title('h')
for aa in ax:
    aa.set_xlabel('x')
plt.tight_layout()
plt.savefig(f'{opath}/fields.png', dpi=300)
plt.close()

# Animate the evolution of the fields
steps = range(0, int(pm.Tevolve/pm.dt) + 1, pm.ostep)
uus, hhs = zip(*(solver.load_fields(opath, step) for step in steps))

def ylim(arrs):
    ''' Common y limits over all frames, with a 5% margin '''
    lo = min(aa.min() for aa in arrs)
    hi = max(aa.max() for aa in arrs)
    pad = 0.05*(hi - lo)
    return lo - pad, hi + pad

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
lu, = ax[0].plot(xx, uus[0])
lh, = ax[1].plot(xx, hhs[0])
ax[0].set_ylim(*ylim(uus))
ax[1].set_ylim(*ylim(hhs))
ax[0].set_ylabel('u')
ax[1].set_ylabel('h')
ax[1].set_xlabel('x')

def update(ii):
    lu.set_ydata(uus[ii])
    lh.set_ydata(hhs[ii])
    ax[0].set_title(f't = {steps[ii]*pm.dt:.2f}')
    return lu, lh

anim = FuncAnimation(fig, update, frames=len(steps))
anim.save(f'{opath}/fields.gif', writer=PillowWriter(fps=20), dpi=80)
plt.close()
