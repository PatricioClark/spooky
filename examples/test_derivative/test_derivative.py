'''
Test derivatives of pseudo-spectral method
'''

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Import corresponding module
import pySPEC.pseudo as ps

# 1D -------------------------------------------------

# Paramters
pm = {"Lx": 22.0, "Nx": 128, "T": 100.0, "dt": 1e-5}
pm = SimpleNamespace(**pm)

# Initialize solver
grid   = ps.Grid1D(pm)

# Function
plt.figure(1)
uu = np.cos(2*np.pi*grid.xx/pm.Lx)
plt.plot(grid.xx, uu)

# Numerical derivative
fu = grid.forward(uu)
du = grid.deriv(fu, grid.kx)
du = grid.inverse(du)
plt.plot(grid.xx, du)

# Analytical derivative
du = -(2*np.pi/pm.Lx)*np.sin(2*np.pi*grid.xx/pm.Lx)
plt.plot(grid.xx, du)

# 2D -------------------------------------------------

# Paramters
pm = {"Lx": 4*np.pi,
      "Nx": 64,
      "Ly": 4*np.pi,
      "Ny": 32,
      "T": 100.0,
      "dt": 1e-5}
pm = SimpleNamespace(**pm)

# Initialize solver
grid   = ps.Grid2D(pm)

# Function
plt.figure(2)
uu = np.cos(2*np.pi*grid.xx/pm.Lx)
j0 = 16
plt.plot(grid.xx[:, j0], uu[:, j0])

# Numerical derivative
fu = grid.forward(uu)
du = grid.deriv(fu, grid.kx)
du = grid.inverse(du)
plt.plot(grid.xx[:, j0], du[:, j0])

# Analytical derivative
du = -(2*np.pi/pm.Lx)*np.sin(2*np.pi*grid.xx/pm.Lx)
plt.plot(grid.xx[:, j0], du[:, j0])

# De-aliasing modes
plt.figure(3)
plt.imshow(grid.dealias_modes)

plt.show()
