If starting new Newton Solver specify input folder and index in params.py, this must contain uu, and vv.
Initial guess for period and shift (T, sx) must be specified in params.py.
If restarting a Newton Solver from a specific Newton iteration enter the iteration in the restart parameter.
- newton_kolmog.py runs the newton solver
- floq_exp.py finds the floquet exponents of a converged orbit (Newton iteration and amount of exponents must be specified inside script)
- plot_kol.py plots relevant quantities such as Newton and GMRes errors, plots orbits, balances, and floquet exponents