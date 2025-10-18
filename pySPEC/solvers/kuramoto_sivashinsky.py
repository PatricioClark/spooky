''' 1D Kuramoto Sivashinsky equation '''

import numpy as np

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class KuramotoSivashinsky(PseudoSpectral):
    ''' 1D Kuramoto Sivashinsky equation '''

    num_fields = 1
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)

        # Define hyperviscosity if not provided
        if not hasattr(self.pm, 'nu'):
            self.pm.nu = 1.0

    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        # Non-linear term
        uu  = self.grid.inverse(fu)
        fu2 = self.grid.forward(uu**2)

        fu = fup + (dt/oo) * (
            - (0.5*1.0j*self.grid.kx*fu2)
            + ((self.grid.k2)*fu)
            - (self.pm.nu*(self.grid.k2**2)*fu)
            )

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0

        return [fu]

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        np.save(f'{opath}uu.{step:0{self.pm.ext}}', uu)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{bpath}balance.dat', 'a') as output:
            print(*bal, file=output)
