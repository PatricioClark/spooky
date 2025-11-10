from dataclasses import dataclass
import os

@dataclass
class Parameters:
    """Container for solver configuration and algorithm parameters."""

    # Input/Output directories (None = disabled)
    input_dir: str = "input"      # Path to input files
    start_idx: int = 0             # Index of input files
    output_dir: str = "output"
    save_outputs: str = "all" # Iterations at which to save evolution outputs. Options: "all", "last", "none" 
    save_balance: str = "all" # Iterations at which to save evolution balance. Options: "all", "last", "none" 
    balance_dir: str|None = "balance"

    # Print directories (None = disabled)
    newton_dir: str|None = "reports"
    gmres_dir: str|None = "reports/gmres"
    apply_A_dir: str|None = "reports/apply_A"
    hookstep_dir: str|None = "reports/hookstep"
    trust_region_dir: str|None = "reports/trust_region"

    # Newton-Solver parameters 
    T: float|None = None          # Initial guess for period. If None, steady states are sought
    Tconst: float|None = None     # If not None, constant evolution time for steady states
    sx: float|None = None         # Initial guess for shift in x. If None then RPOs or TW are not searched for
    N_newt: int = 50              # Maximum number of Newton iterations
    tol_newt: float = 1e-10       # Tolerance for Newton method
    tol_improve: float = 1e-4     # Tolerance for relative improvement of residual
    tol_nudge: float = 1e-3       # Tolerance for selecting different initial condition from orbit if solution is not converging
    frac_nudge: float = 0.        # Fraction of period to nudge initial condition
    restart_iN: int = 0           # Last completed Newton iteration if restarting
    remove_boundary: bool = False # removes boundary components from fields for Krylov calculations 

    # GMRES
    N_gmres: int = 50           # Maximum number of GMRES iterations
    tol_gmres: float = 1e-6     # Tolerance for GMRES method
    eps0: float = 1e-6          # Perturbation factor for Arnoldi iteration

    # Hookstep
    N_hook: int = 10            # Maximum number of hookstep iterations
    c: float = 0.5              # Parameter for regularization condition
    reduc_reg: float = 0.5      # Reduction factor of trust region
    mu0: float = 1e-6           # Initial regularization parameter
    mu_inc: float = 1.5         # Increase factor for regularization

    # Solenoidal projections
    sp1: bool = False        # Performs solenoidal projection in GMRes perturbation
    sp2: bool = False        # Performs solenoidal projection in Newton perturbation
    sp_dU: bool = False      # Performs solenoidal projection over dU instead of U+dU    

    # Arclength Continuation
    arclength: bool = False           # Whether to use arclength continuation
    N_arc: int = 0                    # Maximum number of arclength iterations
    restart_iA: int = 0               # Last running arclength iteration if restarting
    T0: float|None   = None           # Previously converged period. If None, steady states are sought
    sx0: float|None  = None           # Previously converged shift in x. If None then RPOs or TW are not searched for
    lda0: float|None = None           # Previously converged lda.
    converged_dir: str = "converged"  # Directory to save/load converged solutions

    def validate(self):
        """Basic parameter consistency checks."""
        assert os.path.exists(self.input_dir), f"Input directory {self.input_dir} does not exist."
        if self.T is None:
            assert self.Tconst is not None, "If T is None, Tconst must be provided for steady states."
        if self.arclength:
            assert self.lda0 is not None, "lda0 must be provided for arclength continuation."