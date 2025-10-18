from .krylov import GMRES, backsub, arnoldi_eig
import os
import numpy as np

class DynSys():
    def __init__(self, pm, solver):
        '''
        Parameters:
        ----------
            pm: parameters dictionary
            solver: solver object
        '''
        self.pm = pm
        self.solver = solver
        self.grid = solver.grid

    def load_ic(self):
        ''' Load initial conditions '''
        if not self.pm.restart_iN:
            # Start Newton Solver from initial guess
            fields = self.solver.load_fields(self.pm.input, self.pm.stat)

            T, sx = self.pm.T, self.pm.sx
            # Create directories
            self.mkdirs()
            self.write_header()
        else:
            # Restart Newton Solver from last iteration
            restart_path = f'output/iN{self.pm.restart_iN:02}/'
            fields = self.solver.load_fields(restart_path, 0)
            T, sx = self.get_restart_values(self.pm.restart_iN)

        U = self.flatten(fields)
        X = self.form_X(U, T, sx)
        return X

    def form_X(self, U, T=None, sx=None, lda = None):
        ''' Form X vector from fields U, T, sx and lda (if applicable) '''
        X = np.copy(U)
        if self.pm.T is not None:
            X = np.append(X, T)
        if self.pm.sx is not None:
            X = np.append(X, sx)
        if lda is not None:
            X = np.append(X, lda)
        return X

    def unpack_X(self, X, arclength = False):
        '''X could contain extra params sx and lda if searching for RPOs (pm.sx != 0) or using arclength continuation'''

        # Determine size of U
        if getattr(self.pm, 'remove_boundary', False):
            dim_U = self.pm.Nx * (self.pm.Nz-2) * self.solver.num_fields
        else:
            dim_U = self.grid.N * self.solver.num_fields

        U = X[:dim_U]
        idx = dim_U

        if self.pm.T is not None:
            T = X[idx] # T saved as first argument after U
            idx += 1
        else:
            T = self.pm.Tconst # if searching for TW or equilibrium T must be fixed at a small but not too small value

        sx = X[idx] if (self.pm.sx is not None) else 0.

        if not arclength:
            return U, T, sx
        else:
            lda = X[-1]
            return U, T, sx, lda

    def get_restart_values(self, restart, arclength = False, iA:int|None = None ):
        ''' Get values from last Newton iteration of T and sx from solver.txt '''
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''
        fname = f'prints{suffix}/solver.txt'
        iters,T,sx = np.loadtxt(fname, delimiter = ',', skiprows = 1, unpack = True, usecols = (0,2,3))
        idx_restart = np.argwhere(iters==restart)[0][0] #in case more than 1 restart is needed find row with last iter
        values =  [T[idx_restart], sx[idx_restart]]

        if arclength:
            lda = np.loadtxt(fname, delimiter = ',', skiprows = 1, unpack = True, usecols = 5)
            values.append(lda[idx_restart])
        return values

    def flatten(self, fields):
        '''Flattens fields'''
        remove_boundary = self.pm.remove_boundary if hasattr(self.pm, 'remove_boundary') else False
        if not remove_boundary:
            return np.concatenate([f.flatten() for f in fields])
        else:
            return np.concatenate([f[:,1:-1].flatten() for f in fields])

    def unflatten(self, U):
        '''Unflatten fields'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        remove_boundary = self.pm.remove_boundary if hasattr(self.pm, 'remove_boundary') else False
        if not remove_boundary:
            fields = [f.reshape(self.grid.shape) for f in fields]
            return fields
        else:
            trim_shape = (self.pm.Nx, self.pm.Nz-2)
            trim_fields = [f.reshape(trim_shape) for f in fields]
            fields = [np.pad(f, pad_width = ((0,0),(1,1))) for f in trim_fields] #pads second dimension with zeros before and after
            return fields

    def flatten_dec(func):
        """Decorator that allows to work with flattened fields U instead of fields"""
        def wrapper(self, U, *args, **kwargs):
            fields = self.unflatten(U)
            result = func(self, fields, *args, **kwargs)
            return self.flatten(result)
        return wrapper

    @flatten_dec
    def evolve(self, U, T, save = False, iN = 0, iA:int|None = None):
        '''Evolves fields U in time T'''
        if save:
            suffix = f'{iA:02}' if iA is not None else ''
            dir_iN = f'iN{iN:02}/'
            return self.solver.evolve(U, T, bstep = self.pm.bstep, ostep = self.pm.ostep,\
                bpath = f'balance{suffix}/{dir_iN}', opath = f'output{suffix}/{dir_iN}')
        else:
            return self.solver.evolve(U, T)

    @flatten_dec
    def translate(self, U, sx):
        '''Translates fields by sx in x direction'''
        return self.grid.translate(U, sx)

    @flatten_dec
    def deriv_U(self, U, ki):
        '''Derivatives in x direction of fields'''
        return self.grid.deriv_fields(U, ki)

    @flatten_dec
    def sol_project(self, U):
        '''Solenoidal projection of fields'''
        return self.solver.inc_proj(U)

    def write_fields(self, U, idx, path):
        ''' Writes fields to binary file'''
        fields = self.unflatten(U)
        return self.solver.write_fields(fields, idx, path)

    def apply_proj(self, U, dU, sp):
        '''Applies projection if sp is True. To dU or U+dU'''
        if not sp:
            return U+dU

        if self.pm.sp_dU:
            dU = self.sol_project(dU)
            return U+dU
        else:
            return self.sol_project(U+dU)

    def form_b(self, U, UT):
        "Form RHS of extended Newton system. UT is evolved and translated flattened field"
        b = U - UT
        if self.pm.sx is not None:
            b = np.append(b, 0.)
        if self.pm.T is not None:
            b = np.append(b, 0.)
        return b

    def norm(self, U):
        return np.linalg.norm(U)

    def update_A(self, X, iN):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx = self.unpack_X(X)

        # Evolve fields and save output
        self.mkdirs_iN(iN-1) # save output and bal from previous iN
        UT = self.evolve(U, T, save = True, iN = iN-1)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        if self.pm.T is not None:
            dUT_dT = self.evolve(UT, self.pm.dt)
            dUT_dT = (dUT_dT - UT)/self.pm.dt

            dU_dt = self.evolve(U, self.pm.dt)
            dU_dt = (dU_dt - U)/self.pm.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U) # No evol if T is None


        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds = self.unpack_X(dX)

            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT
            if self.pm.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.pm.sx is not None:
                LHS = np.append(LHS, Tx_proj)

            return LHS

        return apply_A, UT

    def run_newton(self, X):
        '''Iterates Newton-GMRes solver until convergence'''
        for iN in range(self.pm.restart_iN+1, self.pm.N_newt):
            # Unpack X
            U, T, sx = self.unpack_X(X)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A(X, iN)

            # RHS of Newton extended system
            b = self.form_b(U, UT)
            F = self.norm(b) #||b|| = ||F||: rootsearch function

            # Write to txts
            self.write_prints(iN, F, U, sx, T)

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, self.pm.N_gmres, self.pm.tol_gmres, iN, self.pm.glob_method)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)

            # Update solution
            U, T, sx = self.unpack_X(X)

            # Select different initial condition from orbit if solution is not converging
            if ((F-F_new)/F) < self.pm.tol_nudge:
                with open('prints/nudge.txt', 'a') as file:
                    file.write(f'iN = {iN}. frac_nudge = {self.pm.frac_nudge}\n')
                U = self.evolve(U, T*self.pm.frac_nudge)

            # Termination condition
            if (F_new < self.pm.tol_newt) and ((F-F_new)/F < self.pm.tol_improve):

                self.mkdirs_iN(iN)
                UT = self.evolve(U, T, save = True, iN = iN)
                if self.pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                F = self.norm(b) #||b|| = ||F||: rootsearch function
                # Write to txts
                self.write_prints(iN+1, F, U, sx, T)
                break

    def hookstep(self, X, H, beta, Q, iN, arclength:dict|None = None):
        ''' Performs hookstep on solution given by GMRes untill new |F| is less than previous |F| (or max iter of hookstep is reached) '''
        ''' If performing arclength continuation arclength contains a dict with the relevant quantities '''
        # Unpack X
        if not arclength:
            U, T, sx = self.unpack_X(X)
        else:
            U, T, sx, lda = self.unpack_X(X, arclength = True)

        #Initial solution from GMRes in basis Q
        y = backsub(H, beta)
        #Initial trust region radius
        Delta = self.norm(y)

        #Define trust_region function
        if not arclength:
            trust_region = self.trust_region_function(H, beta, iN, y)
        else:
            trust_region = self.trust_region_function(H, beta, iN, y, arclength["iA"])

        mu = 0.
        #Perform hookstep
        for iH in range(self.pm.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y #Unitary transform back to full dimension

            if not arclength:
                dU, dT, dsx = self.unpack_X(dx)
            else:
                dU, dT, dsx, dlda = self.unpack_X(dx, arclength = True)

            U_new = self.apply_proj(U, dU, self.pm.sp2)

            if self.pm.sx is not None:
                sx_new = sx+dsx.real
            else:
                sx_new = 0.

            if self.pm.T:
                T_new = T+dT.real
            else:
                T_new = self.pm.Tconst

            X_new = self.form_X(U_new, T_new, sx_new)

            if arclength:
                lda_new = lda+dlda.real
                X_new = np.append(X_new, lda_new)
                self.update_lda(lda_new)

            UT = self.evolve(U_new, T_new)
            if self.pm.sx is not None:
                UT = self.translate(UT,sx_new)

            b = self.form_b(U_new, UT)
            if arclength:
                # add arclength term
                N = self.N_constraint(X_new, arclength["dX_dr"], arclength["dr"], arclength["X1"])
                b = np.append(b, -N)

            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.pm.c * H @ y) #linear expansion of F around x (in basis Q).
            # beta = H@y holds

            if arclength:
                self.write_hookstep(iN, iH, F_new, lin_exp, arclength["iA"])
            else:
                self.write_hookstep(iN, iH, F_new, lin_exp)

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.pm.reduc_reg #reduce trust region
        return X_new, F_new, UT


    def trust_region_function(self, H, beta, iN, y0, iA:int|None =None):
        ''' Performs trust region on solution provided by GMRes. Must be instantiated at each Newton iteration '''
        R = H #R part of QR decomposition
        A_ = R.T @ R #A matrix in trust region
        b = R.T @ beta #b vector in trust region

        def trust_region(Delta, mu):
            ''' Delta: trust region radius, mu: penalty parameter  '''
            if mu == 0: #First hoostep iteration. Doens't perform trust region
                y_norm = self.norm(y0)
                Delta0 = y_norm #Initial trust region

                suffix = f'{iA:02}' if iA is not None else ''
                with open(f'prints{suffix}/hookstep/extra_iN{iN:02}.txt', 'a') as file:
                    file.write(f'{Delta0},{mu},{y_norm},0\n')

                mu = self.pm.mu0 #Initialize first nonzero value of mu
                return y0, mu

            for _ in range(1, 1000): #1000 big enough to ensure that condition is satisfied

                # Ridge regression adjustment
                A = A_ + mu * np.eye(A_.shape[0])
                y = np.linalg.solve(A, b) #updates solution with trust region

                self.write_trust_region(iN, Delta, mu, y, A, iA)

                if self.norm(y) <= Delta:
                    break
                else:
                    # Increase mu until trust region is satisfied
                    mu *= self.pm.mu_inc

            return y, mu
        return trust_region

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def mkdirs(self, iA:int | None = None):
        ''' Make directories for solver
        if iA is given then outputs are saved in a specific arclength iteration'''

        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        dirs = ['output', 'balance', 'prints']
        dirs = [dir_+ suffix for dir_ in dirs]

        prints_dir = dirs[-1]
        dirs.extend ([prints_dir+dir_ for dir_ in ('/error_gmres', '/hookstep', '/apply_A')])

        if self.solver.solver == 'BOUSS' or self.solver.solver == 'ROTH':
            dirs.append('bin_tmp')

        for dir_ in dirs:
            self.mkdir(dir_)

    def mkdirs_iN(self, iN, iA:int | None = None):
        ''' Directories for specific Newton iteration
        if iA is given then outputs are saved in a specific arclength iteration'''

        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        dirs = [f'output{suffix}/iN{iN:02}', f'balance{suffix}/iN{iN:02}']

        for dir_ in dirs:
            self.mkdir(dir_)

    def write_header(self, arclength = False, iA:int|None=None):
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        with open(f'prints{suffix}/solver.txt', 'a') as file:
            file.write('Iter newt, |F|, T, sx, |U|')
            if arclength:
                file.write(', lambda, N(X)')
            file.write('\n')

    def write_prints(self, iN, F, U, sx, T, arclength = None, iA:int|None = None):
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        with open(f'prints{suffix}/solver.txt', 'a') as file1,\
        open(f'prints{suffix}/error_gmres/iN{iN:02}.txt', 'w') as file2,\
        open(f'prints{suffix}/hookstep/iN{iN:02}.txt', 'w') as file3,\
        open(f'prints{suffix}/apply_A/iN{iN:02}.txt', 'w') as file4,\
        open(f'prints{suffix}/hookstep/extra_iN{iN:02}.txt', 'w') as file5:

            file1.write(f'{iN-1:02},{F:.6e},{T},{sx:.8e},{self.norm(U):.6e}')
            if arclength:
                file1.write(f',{arclength[0]},{arclength[1]}') #writes lambda (arclength[0]) and N(X) (arclength[1])
            file1.write('\n')
            file2.write('iG, error\n')
            file3.write('iH, |F|, |F(x)+cAdx|, |F(x)+Adx|\n')

            if arclength:
                lda_str = ', |dUT_dlda|'
            else:
                lda_str = ''
            file4.write(f'|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|{lda_str}, t_proj, Tx_proj')
            if arclength:
                file4.write(f', lda_proj')
            file4.write('\n')
            file5.write('Delta, mu, |y|, cond(A)\n')

    def write_apply_A(self, iN, norms, t_proj, Tx_proj, lda_proj = None, iA:int|None = None):
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        with open(f'prints{suffix}/apply_A/iN{iN:02}.txt', 'a') as file:
            file.write(','.join([f'{norm:.4e}' for norm in norms]) + f',{t_proj:.4e},{Tx_proj:.4e}')
            if lda_proj:
                file.write(f',{lda_proj:.4e}')
            file.write('\n')

    def write_trust_region(self, iN, Delta, mu, y, A, iA:int|None = None):
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        with open(f'prints{suffix}/hookstep/extra_iN{iN:02}.txt', 'a') as file:
            file.write(f'{Delta:.4e},{mu:.4e},{self.norm(y):.4e},{np.linalg.cond(A):.3e}\n')

    def write_hookstep(self, iN, iH, F_new, lin_exp, iA:int|None = None):
        # Convert iA to a string prefix if it's provided
        suffix = f'{iA:02}' if iA is not None else ''

        with open(f'prints{suffix}/hookstep/iN{iN:02}.txt', 'a') as file:
            file.write(f'{iH:02},{F_new:.4e},{lin_exp:.4e}\n')

    def phase_shifted_b(self, fields):
        """Generate b vector by modifying phases in Fourier space."""
        def apply_phase_shift(U):
            """Applies a random phase shift to a single field U and projects to solenoidal modes."""
            U_hat = self.grid.forward(U)
            random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, U_hat.shape))
            U_hat_shifted = np.abs(U_hat) * random_phases  # Preserve magnitudes
            return U_hat_shifted

        # Separate into u and v components
        u, v = fields
        b_vector = []

        u_hat_shifted = apply_phase_shift(u)
        v_hat_shifted = apply_phase_shift(v)

        # Project in Fourier space
        u_proj_hat, v_proj_hat = self.grid.inc_proj((u_hat_shifted, v_hat_shifted))

        # Inverse FFT to physical space
        u_proj = self.grid.inverse(u_proj_hat)
        v_proj = self.grid.inverse(v_proj_hat)
        fields_proj = [u_proj, v_proj]

        b_vector = self.flatten([f.flatten() for f in fields_proj])
        return b_vector


    def floq_exp(self, X, n, tol, b = 'U'):
        ''' Calculates Floquet exponents of periodic orbit '''
        ''' X: (U,T,sx) of converged periodic orbit, n: number of exponents, tol: tolerance of Arnoldi '''

        from warnings import warn
        warn('This function is deprecated. Use floquet_exponents instead.',
             DeprecationWarning)

        # Unpack X
        U, T, sx = self.unpack_X(X)

        UT = self.evolve(U, T)
        # Translate UT by sx
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU
            U_pert = U + epsilon*dU

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon
            return dUT_dU

        if b == 'U':
            b = U
        elif b == 'random':
            b = np.random.randn(len(U))
        else:
            raise ValueError('b must be U or random')

        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        return eigval_H, eigvec_H, Q

    def lyap_exp(self, fields, T, n, tol, ep0=1e-7, sx=None, b='U'):
        from warnings import warn
        warn('This function is deprecated. Use floquet_exponents instead.',
             DeprecationWarning)
        return self.floquet_exponents(fields, T, n, tol, ep0, sx, b)

    def floquet_exponents(self, fields, T, n, tol, ep0=1e-7, sx=None, b='U', test=False):
        ''' Calculates Floquet exponents

        To get the Lyapunov exponents do log(eigval_H)/T

        Paramters
        ---------
        fields: list of fields.
        T: time to evolve in each arnoldi iteration
        n: number of exponents,
        tol: tolerance of Arnoldi
        ep0: perturbation factor, optional, default=1e-7
        sx: translation in x direction, optional, default=None
        b: initial guess for Arnoldi, optional, default='U'. Could also take in 'random'
        or a np.array of the same size as U

        Returns
        -------
        eigval_H: Lyapunov exponents
        eigvec_H: Lyapunov vectors
        Q: Arnoldi basis
        '''

        U = self.flatten(fields)
        UT = self.evolve(U, T)

        # Translate UT by sx
        if sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = ep0*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU
            U_pert = U + epsilon*dU

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)

            if sx is not None:
                dUT_dU = self.translate(dUT_dU, sx)

            dUT_dU = (dUT_dU - UT)/epsilon
            return dUT_dU

        if isinstance(b, str):
            if b == 'U':
                b = U
            elif b == 'random':
                b = np.random.randn(len(U))
            elif b == 'phases':
                b = self.phase_shifted_b(fields)
        elif isinstance(b, np.ndarray):
            pass
        else:
            raise ValueError('b must be one of the given options')

        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        if test:
            # Checks if returned eigenvectors satisfy Av=lambda v
            Aev_r = np.zeros((len(U), n))
            Aev_i = np.zeros((len(U), n))
            for i in range(n):
                ev = Q @ eigvec_H[:,i]
                Aev_r[:,i] = apply_J(ev.real)
                Aev_i[:,i] = apply_J(ev.imag)

            return eigval_H, eigvec_H, Q, Aev_r, Aev_i

        return eigval_H, eigvec_H, Q

    def run_arclength(self, X0, X1, X_restart = None, iA:int|None = None):
        '''Iterates Newton-GMRes solver until convergence using arclength continuation
        Follows convention of Chandler - Kerswell: Invariant recurrent solutions..
        X: Vector containing (U, T, sx, lda) to be updated with Newton until it converges to periodic orbit
        dX_dr: Derivative of X w.r.t. arclength
        X1: Previous converged solution (X(r0) in Chandler-Kerswell
        iA: arclength iteration in case automatic arclength is performed '''

        # Define relevant quantities for arclength continuation
        dr = np.linalg.norm(X1 - X0)
        dX_dr = (X1 - X0)/dr

        if X_restart is not None: # in case Newton iteration is to be restarted
            X = X_restart
        else:
            X = X1

        start_iN = self.pm.restart_iN+1 if (iA==self.pm.restart_iA) else 1

        for iN in range(start_iN, self.pm.N_newt):
            # lda: lambda parameter, Re (Reynolds) if solver=='KolmogorovFlow', Ra (Rayleigh) if solver=='BOUSS'
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A_arc(X, dX_dr, iN, iA)

            # RHS of Newton extended system
            # form b (rhs)
            b = self.form_b(U, UT)

            # add arclength term
            N = self.N_constraint(X, dX_dr, dr, X1)
            b = np.append(b, -N)

            F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint

            # Write to txts
            self.write_prints(iN, F, U, sx, T, (lda*self.pm.norm, N), iA)

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, self.pm.N_gmres, self.pm.tol_gmres, iN, self.pm.glob_method, iA)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN, arclength = {'iA':iA, 'dX_dr':dX_dr, 'dr':dr, 'X1':X1})

            # Update solution
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Termination condition
            if (F_new < self.pm.tol_newt) and ((F-F_new)/F < self.pm.tol_improve):
                self.mkdirs_iN(iN, iA)
                UT = self.evolve(U, T, save = True, iN = iN, iA = iA)
                if self.pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                # add arclength term
                N = self.N_constraint(X, dX_dr, dr, X1)
                b = np.append(b, -N)
                F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint
                # Write to txts
                self.write_prints(iN+1, F, U, sx, T, (lda*self.pm.norm, N), iA)

                if iA is not None:
                    self.save_converged(X, iA)
                return X

    def save_converged(self, X, iA):
        U, T, sx, lda = self.unpack_X(X, arclength = True)
        path = f'converged/iA{iA:02}'
        self.mkdir(path)
        if iA == 0:
            print('iA, T, sx, Ra, |U|', file= open('converged/solver.txt', 'a'))

        print(f'{iA:02},{T},{sx:.8e},{lda*self.pm.norm},{self.norm(U):.6e}',\
        file = open('converged/solver.txt', 'a'))
        self.write_fields(U, idx = 0, path = path)

    def update_lda(self, lda):
        '''Updates lambda parameter in solver. norm: normalization parameter'''
        if self.solver.solver == 'KolmogorovFlow':
            Re = lda # Reynolds number
            self.pm.nu = 1 / Re
        elif self.solver.solver == 'BOUSS':
            Ra = lda # Rayleigh number
            Ra *= self.pm.norm # to account for possible normalization
            self.pm.ra = Ra
        else:
            raise Exception('Arclength only implemented for Kolmogorov and BOUSS')

    def N_constraint(self, X, dX_dr, dr, X1):
        '''Calculates N function resulting from the arclength constraint as in
        'Chandler - Kerswell: Invariant recurrent solutions..' '''
        alpha = getattr(self.pm, 'alpha', 1.) # parametrization velocity. default = 1.
        return np.dot(dX_dr, (X-X1)) - dr * alpha**2

    def update_A_arc(self, X, dX_dr, iN, iA: int|None):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx, lda = self.unpack_X(X, arclength = True)

        # Update lambda
        self.update_lda(lda)

        # Evolve fields and save output
        self.mkdirs_iN(iN-1, iA) # save output and bal from previous iN
        UT = self.evolve(U, T, save = True, iN = iN-1, iA = iA)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        if self.pm.T is not None:
            dUT_dT = self.evolve(UT, self.pm.dt)
            dUT_dT = (dUT_dT - UT)/self.pm.dt

            dU_dt = self.evolve(U, self.pm.dt)
            dU_dt = (dU_dt - U)/self.pm.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U) # No evol if T is None

        # Calculate derivative of evolved translated fields wrt lambda
        dlda = lda * 1e-3
        self.update_lda(lda + dlda)
        dUT_dlda = self.evolve(U, T)
        dUT_dlda = self.translate(dUT_dlda,sx)
        dUT_dlda = (dUT_dlda - UT)/dlda

        # Return lambda to previous state
        self.update_lda(lda)

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds, dlda = self.unpack_X(dX, arclength = True)

            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real
            lda_proj = np.dot(dX_dr.conj(), dX).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, dUT_dlda]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj, lda_proj, iA)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT + dUT_dlda*dlda
            if self.pm.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.pm.sx is not None:
                LHS = np.append(LHS, Tx_proj)
            # Add arclength term
            LHS = np.append(LHS, lda_proj)

            return LHS

        return apply_A, UT


    def run_arc_auto(self, X0, X1, X_restart = None):
        '''Iterates Newton-GMRes solver until convergence using arclength continuation
        Once a solution has converged it automatically uses the new converged solution as
        the new starting point
        '''
        start_iA = max(2, self.pm.restart_iA)

        for iA in range(start_iA, self.pm.N_arc):
            if (self.pm.restart_iN == 0) or (iA != start_iA):
                self.mkdirs(iA)
                self.write_header(arclength = True, iA = iA)

            X = self.run_arclength(X0, X1, X_restart, iA)
            X0 = X1
            X1 = X
            X_restart = None
