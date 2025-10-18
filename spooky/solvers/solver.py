import abc
import numpy as np

from .. import pseudo as ps

class Solver(abc.ABC):
    ''' Abstract solver class

    All solvers must implement the following methods:
        - evolve: Evolve solver method
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs

    All solvers have defined the following: 
        - num_fields: Number of fields
        - dim_fields: Spatial dimension of the fields
    '''
    num_fields: int
    dim_fields: int

    def __init__(self, pm):
        ''' Initializes the solver

        Parameters:
        ----------
            pm: parameter dictionary
        '''
        self.pm = pm
        self.pm.dim = self.dim_fields
        self.grid: ps.Grid1D | ps.Grid2D | ps.Grid2D_semi | ps.Grid3D

    @abc.abstractmethod
    def evolve(self, fields, T, bstep=None, ostep=None, sstep=None, bpath = '', opath = '', spath = ''):
        return []

    def balance(self, fields, step, bpath):
        pass

    def spectra(self, fields, step, spath):
        pass

    def outs(self, fields, step, opath):
        pass

    def write_outputs(self, fields, step, bstep, ostep, sstep, bpath, opath, spath, final = False):
        if bstep is not None and (step%bstep==0 or final):
            self.balance(fields, step, bpath)

        if sstep is not None and (step%sstep==0 or final):
            self.spectra(fields, step, spath)
            
        if ostep is not None and (step%ostep==0 or final):
            self.outs(fields, step, opath)
