# GHOST parameters (set as compiled solver) 
import numpy as np
Lx = 2*np.pi # Domain length in x
Ly = 2*np.pi # Domain length in x
Nx = 256 # Number of grid points in x
Ny = 256 # Number of grid points in y
precision = 'double' # Precision of the code
ext = 5 # Number of digits in file names

# GHOST adjustable parameters
Tevolve = 1.0 # Evolve time
nu = 1./40. # Kinematic viscosity
dt = 1e-3 # Time step
bstep = 100 # Time step for saving text files
ostep = 200 # Time step for saving output files
sstep = 0 # Time step for saving spectra
opath = 'output' # Output path
nprocs = 10 # Number of processors
