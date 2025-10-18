# GHOST parameters (set as compiled solver) 
L = 1.0 # Domain length in x (multiples of 2pi)
Nx = 256 # Number of grid points in x
Ny = 256 # Number of grid points in y
precision = 'double' # Precision of the code
ext = 5 # Number of digits in file names

# GHOST adjustable parameters
T = 1.0 # Evolve time
nu = 1./40. # Kinematic viscosity
dt = 1e-3 # Time step
bstep = 100 # Time step for saving text files
ostep = 200 # Time step for saving output files
sstep = 0 # Time step for saving spectra
ext = 5 # Number of digits in file names
ipath = 'outkol_ghost' # Input path
opath = 'outkol_ghost' # Output path
bpath = 'outkol_ghost' # Balance path
spath = 'spectra' # Spectra path
nprocs = 10 # Number of processors