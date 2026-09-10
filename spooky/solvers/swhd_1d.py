''' 1D Shallow Water Equations '''

import numpy as np
import os
from .._backend import xnp, index_update, apply_jit

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps


class SWHD_1D(PseudoSpectral):
    ''' 1D Shallow Water Equations with periodic boundary conditions

        u_t + u u_x + g h_x = 0
        h_t + ( u (h - hb) )_x = 0

    where u is the velocity, h the free surface height and hb the bottom
    topography, so that h - hb is the fluid depth.

    The zero mode of u is set to zero after every RK sub-step (the mean
    velocity is conserved by the equations, so this only fixes the frame).
    The zero mode of h is kept, since the mean height is not null.

    Parameters
    ---------
    grid: Grid1D instance
        Defines spatiotemporal grid parameters.
    g: float, optional
        Gravity. Default is 1.0.
    hb: array_like, optional
        Bottom topography in physical space, shape (Nx,). Default is a flat
        bottom, hb = 0.
    rkord: int, optional
        Order of the RK integration. Default is 2.
    ext: int, optional
        Length of zero padding in output file name. Default is 4.
    '''

    num_fields = 2
    dim_fields = 1

    def __init__(self,
                 grid: ps.Grid1D,
                 g=1.0,
                 hb=None,
                 rkord=2,
                 ext=4):
        super().__init__(grid, rkord=rkord)
        self.g = g
        self.ext = ext

        if hb is None:
            hb = xnp.zeros(grid.shape)
        self.hb = xnp.asarray(hb)

    @apply_jit
    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fu, fh = fields
        fup, fhp = prev

        # Non-linear terms
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)
        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))

        fu_ux = self.grid.forward(uu*ux)
        fuH_x = self.grid.deriv(self.grid.forward(uu*(hh - self.hb)), self.grid.kx)
        fhx   = self.grid.deriv(fh, self.grid.kx)

        # Equations
        fu = fup + (dt/oo) * (- fu_ux - self.g*fhx)
        fh = fhp + (dt/oo) * (- fuH_x)

        # de-aliasing
        fu = index_update(fu, self.grid.zero_mode, 0.0)
        fu = index_update(fu, self.grid.dealias_modes, 0.0)
        fh = index_update(fh, self.grid.dealias_modes, 0.0)

        return [fu, fh]

    def mass(self, fields):
        ''' Mean fluid depth <h - hb> (conserved) '''
        hh = self.grid.inverse(fields[1])
        return xnp.mean(hh - self.hb)

    def energy(self, fields):
        ''' Mean energy density <(h - hb) u^2 / 2 + g h^2 / 2> (conserved) '''
        uu = self.grid.inverse(fields[0])
        hh = self.grid.inverse(fields[1])
        return xnp.mean(0.5*(hh - self.hb)*uu**2 + 0.5*self.g*hh**2)

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        hh = self.grid.inverse(fields[1])
        np.save(os.path.join(opath, f'uu.{step:0{self.ext}}'), uu)
        np.save(os.path.join(opath, f'hh.{step:0{self.ext}}'), hh)

    def balance(self, fields, step, bpath):
        mass = self.mass(fields)
        eng  = self.energy(fields)

        bal = [f'{self.grid.dt*step:.4e}', f'{mass:.6e}', f'{eng:.6e}']
        with open(os.path.join(bpath, 'balance.dat'), 'a') as output:
            print(*bal, file=output)

    def load_fields(self, path, step, ext=None):
        if not ext:
            ext = self.ext
        uu = np.load(os.path.join(path, f'uu.{step:0{ext}}.npy'))
        hh = np.load(os.path.join(path, f'hh.{step:0{ext}}.npy'))
        return [uu, hh]
