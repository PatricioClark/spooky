import abc
import numpy as np

from .solver import Solver
from .. import pseudo as ps

class PseudoSpectral(Solver, abc.ABC):
    ''' Abstract Pseudospectral solver class

    All solvers must implement the following methods:
        - rkstep: Runge-Kutta step
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs
    '''
    def __init__(self, pm):
        super().__init__(pm)

    def evolve(self, fields, T, bstep=None, ostep=None, sstep=None, bpath = '', opath = '', spath = '', write_outputs=True):
        ''' Evolves velocity fields to time T '''

        # Forward transform
        fields = [self.grid.forward(ff) for ff in fields]

        Nt = int(T/self.pm.dt)
        for step in range(Nt+1):
            # Store previous time step
            prev = np.copy(fields)

            # Write outputs
            if write_outputs:
                self.write_outputs(fields, step, bstep, ostep, sstep, bpath, opath, spath)

            dt = self.grid.dt
            #Make exact last step
            if step == Nt:
                dt = T - Nt*self.grid.dt
                if np.isclose(dt, 0., atol=1e-8):
                    break

            # Time integration
            for oo in range(self.pm.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo, dt)

        # Write final outputs
        if write_outputs:
            self.write_outputs(fields, Nt, bstep, ostep, sstep, bpath, opath, spath, final=True)

        # Inverse transform
        fields = [self.grid.inverse(ff) for ff in fields]
        return fields

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo, dt):
        return []
