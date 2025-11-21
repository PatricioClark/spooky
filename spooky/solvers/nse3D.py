''' 3D Navier-Stokes solver '''

import numpy as np
from .._backend import xnp, index_update, apply_jit
import os

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps


class NSE3D(PseudoSpectral):
    '''
    NSE 3D

    nu = 1/Re

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.
    '''

    num_fields = 3
    dim_fields = 3

    def __init__(self, pm, kf=4, f0=1., ftypes=['uu', 'vv', 'ww']):
        super().__init__(pm)
        self.grid = ps.Grid3D(pm)
        self.ftypes = ftypes
        self.solver = 'NSE3D'

        # Forcing
        self.kf = kf
        self.f0 = f0
        self.fx = f0*xnp.sin(2*np.pi*kf*self.grid.yy/pm.Ly)
        self.fx = self.grid.forward(self.fx)
        self.fy = xnp.zeros_like(self.fx, dtype=complex)
        self.fz = xnp.zeros_like(self.fx, dtype=complex)
        self.fx, self.fy, self.fz = self.grid.inc_proj([self.fx, self.fy, self.fz])

    @apply_jit
    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fu, fv, fw = fields
        fup, fvp, fwp = prev

        # Non-linear term
        uu = self.grid.inverse(fu)
        vv = self.grid.inverse(fv)
        ww = self.grid.inverse(fw)
        
        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))
        uy = self.grid.inverse(self.grid.deriv(fu, self.grid.ky))
        uz = self.grid.inverse(self.grid.deriv(fu, self.grid.kz))

        vx = self.grid.inverse(self.grid.deriv(fv, self.grid.kx))
        vy = self.grid.inverse(self.grid.deriv(fv, self.grid.ky))
        vz = self.grid.inverse(self.grid.deriv(fv, self.grid.kz))

        wx = self.grid.inverse(self.grid.deriv(fw, self.grid.kx))
        wy = self.grid.inverse(self.grid.deriv(fw, self.grid.ky))
        wz = self.grid.inverse(self.grid.deriv(fw, self.grid.kz))

        gx = self.grid.forward(uu*ux + vv*uy + ww*uz)
        gy = self.grid.forward(uu*vx + vv*vy + ww*vz)
        gz = self.grid.forward(uu*wx + vv*wy + ww*wz)
        gx, gy, gz = self.grid.inc_proj([gx, gy, gz])

        # Equations
        fu = fup + (dt/oo) * (
            - gx
            - self.pm.nu * self.grid.k2 * fu 
            + self.fx
            )

        fv = fvp + (dt/oo) * (
            - gy
            - self.pm.nu * self.grid.k2 * fv 
            + self.fy
            )

        fw = fwp + (dt/oo) * (
            - gz
            - self.pm.nu * self.grid.k2 * fw 
            + self.fz
            )

        # de-aliasing
        fu = fu.at[self.grid.zero_mode].set(0.0)
        fu = fu.at[self.grid.dealias_modes].set(0.0)
        fw = fw.at[self.grid.dealias_modes].set(0.0)
        fv = fv.at[self.grid.zero_mode].set(0.0)
        fv = fv.at[self.grid.dealias_modes].set(0.0)
        fw = fw.at[self.grid.dealias_modes].set(0.0)

        return [fu, fv, fw]

    def injection(self, fields, forcing):
        return self.grid.avg(self.grid.inner(fields, forcing))

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        vv = self.grid.inverse(fields[1])
        np.save(os.path.join(opath,f'uu.{step:0{self.pm.ext}}'), uu)
        np.save(os.path.join(opath,f'vv.{step:0{self.pm.ext}}'), vv)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        ens = self.grid.enstrophy(fields)
        dis = - 2 * self.pm.nu * ens
        inj = self.injection(fields, [self.fx, self.fy])

        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}', f'{dis:.6e}', f'{inj:.6e}']
        with open(os.path.join(bpath, 'balance.dat'), 'a') as output:
            print(*bal, file=output)

    def load_fields(self, path, step, ext = None):
        if not ext:
            ext = self.pm.ext
        uu = np.load(os.path.join(path, f'uu.{step:0{ext}}.npy'))
        vv = np.load(os.path.join(path, f'vv.{step:0{ext}}.npy'))
        return [uu, vv]

    def oz(self, fields):
        ''' Computes vorticity field '''
        fu, fv = [self.grid.forward(ff) for ff in fields]
        uy = self.grid.inverse(self.grid.deriv(fu, self.grid.ky))
        vx = self.grid.inverse(self.grid.deriv(fv, self.grid.kx))
        return uy - vx

    def inc_proj(self, fields):
        fields = [self.grid.forward(ff) for ff in fields]
        inc_f = self.grid.inc_proj(fields)         
        fields = [self.grid.inverse(ff) for ff in inc_f]
        return fields
