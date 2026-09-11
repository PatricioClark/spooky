import numpy as np

Lx = 2*np.pi         # Domain length in x
Nx = 1024            # Number of grid points in x
Tevolve = 5.0        # Total evolution time
dt = 1e-4            # Time step
g = 5.0              # Gravity
h0 = 0.2             # Mean free surface height
rkord = 2            # Runge-Kutta order
ostep = 250          # Output step
bstep = 250          # Balance step
