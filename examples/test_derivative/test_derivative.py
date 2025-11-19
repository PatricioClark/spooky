'''
Test derivatives of pseudo-spectral method
'''

import numpy as np
import matplotlib.pyplot as plt

# Import corresponding module
import spooky.pseudo as ps

# 1D -------------------------------------------------

# Paramters
Lx = 22.0
Nx = 128
dt = 1e-5

# Initialize solver
grid   = ps.Grid1D(Lx=Lx, Nx=Nx, dt=dt)

# Function
plt.figure(1)
uu = np.cos(2*np.pi*grid.xx/Lx)
plt.plot(grid.xx, uu)

# Numerical derivative
fu = grid.forward(uu)
du = grid.deriv(fu, grid.kx)
du = grid.inverse(du)
plt.plot(grid.xx, du)

# Analytical derivative
du = -(2*np.pi/Lx)*np.sin(2*np.pi*grid.xx/Lx)
plt.plot(grid.xx, du, '--')

# 2D -------------------------------------------------

# Paramters
Lx = 4*np.pi
Ly = 4*np.pi
Nx = 64
Ny = 32
dt = 1e-5

# Initialize solver
grid   = ps.Grid2D(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt=dt)

# Function
plt.figure(2)
uu = np.cos(2*np.pi*grid.xx/Lx)
j0 = 16
plt.plot(grid.xx[:, j0], uu[:, j0])

# Numerical derivative
fu = grid.forward(uu)
du = grid.deriv(fu, grid.kx)
du = grid.inverse(du)
plt.plot(grid.xx[:, j0], du[:, j0])

# Analytical derivative
du = -(2*np.pi/Lx)*np.sin(2*np.pi*grid.xx/Lx)
plt.plot(grid.xx[:, j0], du[:, j0], '--')

# De-aliasing modes
plt.figure(3)
plt.imshow(grid.dealias_modes)

plt.show()
