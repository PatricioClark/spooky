# GHOST parameters (set as compiled solver) 
import numpy as np
Lx = 2*np.pi # Domain length in x
Lz = 3.14159 # Domain length in x
Nx = 256 # Number of grid points in x
Nz = 103 # Number of grid points in y
precision = 'double' # Precision of the code
ext = 5 # Number of digits in file names

# GHOST adjustable parameters
Tevolve = 1.0 # Evolve time
ra = 5e5 # Rayleigh number
pr = 1.0 # Prandtl number
dt = 5e-4 # Time step

bstep = 100 # Time step for saving text files
ostep = 200 # Time step for saving output files
opath = 'output' # Output path
bpath = 'balance' # Balance output path
nprocs = 14 # Number of processors
