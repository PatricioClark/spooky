# GHOST parameters (to be set as compiled solver) 
Lx = 1.0 # Domain length in x (multiples of 2pi)
Ly = 1.0 # Domain length in y (multiples of 2pi)
Lz = 1.0 # Domain length in z (multiples of 2pi)
Nx = 64 # Number of grid points in x
Ny = 64 # Number of grid points in y
Nz = 64 # Number of grid points in z
precision = 'single' # Precision of the code
ext = 4 # Number of digits in file names
rkord = 2 # Order of Runge-Kutta method

# GHOST adjustable parameters
dt = 7.5e-4 # Time step
bstep = 100 # Time step for saving text files
ostep = 200 # Time step for saving output files
nprocs = 14 # Number of processors dedicated to simulation

# Newton-Krylov parameters 
T = None # Initial guess for period. If None then equilibrium is searched for
sx = 0. # Initial guess of shift in x. If None then Traveling Waves are not searched for  
restart_iN = 0 # Last completed Newton iteration if restarting
input = "input/" # Path to input files
stat = 19 # Index of input files 
Tconst = 10. # If searching for Equilibrium (or TW) the fixed time to evolve per iteration

remove_boundary = False  # Removes fixed non-periodic boundaries from fields for Newton Krylov calculations

N_newt = 200 # Maximum number of Newton iterations
N_gmres = 300 # Maximum number of GMRES iterations
tol_newt = 1e-10 # Tolerance for Newton iterations
tol_gmres = 1e-3 # Tolerance for GMRES iterations

glob_method = 1 # Global Newton Method: 0 for no global method, 1 for hookstep
N_hook = 25 # Maximum number of hookstep iterations
c = 0.5 # Parameter for regularization condition
reduc_reg = 0.5 # Reduction factor of trust region
mu0 = 1e-6 # Initial regularization parameter
mu_inc = 1.5 # Increase factor for regularization

sp1 = False # Performs solenoidal projection in GMRes perturbation
sp2 = False # Performs solenoidal projection in Newton perturbation
sp_dU = False # Performs solenoidal projection over dU instead of U+dU
cmplx = False # Use complex velocity fields for Newton solver

tol_nudge = 1e-3 # Tolerance for selecting different initial condition from orbit if solution is not converging
frac_nudge = 0. # Fraction of period to nudge initial condition
