'''
Pseudo-spectral solver for the 1D Kuramoto-Sivashinsky equation
'''

import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import KuramotoSivashinsky

import params as pm

#  Initialize solver
grid   = sp.Grid1D(pm.Lx, pm.Nx, pm.dt)
solver = KuramotoSivashinsky(grid)

# Initial conditions
uu = (0.3*np.cos(2*np.pi*3.0*grid.xx/pm.Lx) +
    0.4*np.cos(2*np.pi*5.0*grid.xx/pm.Lx) +
    0.5*np.cos(2*np.pi*4.0*grid.xx/pm.Lx)
    )
fields = [uu]

# Evolve
fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)

plt.figure()
plt.plot(bal[0], bal[1])
plt.savefig('energy.png', dpi = 300)
plt.close()    
