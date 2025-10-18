'''
Pseudo-spectral solver for the 1D Kuramoto-Sivashinsky equation
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pySPEC as ps
from pySPEC.solvers import KuramotoSivashinsky

# Parse JSON into an object with attributes corresponding to dict keys.
pm = json.load(open('params.json', 'r'), object_hook=lambda d: SimpleNamespace(**d))

# Initialize solver
grid   = ps.Grid1D(pm)
solver = KuramotoSivashinsky(pm)

# Initial conditions
uu = (0.3*np.cos(2*np.pi*3.0*grid.xx/pm.Lx) +
      0.4*np.cos(2*np.pi*5.0*grid.xx/pm.Lx) +
      0.5*np.cos(2*np.pi*4.0*grid.xx/pm.Lx)
      )
fields = [uu]

# Evolve
fields = solver.evolve(fields, pm.T, bstep=pm.bstep, ostep=pm.ostep)

# Plot Balance
bal = np.loadtxt('balance.dat', unpack=True)
plt.plot(bal[0], bal[1])
plt.savefig('balance.png')

# Plot fields
acc = []
for ii in range(0,int(pm.T/pm.dt), pm.ostep):
    out = np.load(f'uu.{ii:04}.npy')
    acc.append(out)

acc = np.array(acc)
plt.figure()
plt.imshow(acc, extent=[0,pm.Lx,0,pm.T], aspect='auto')
plt.savefig('acc.png')
plt.show()
