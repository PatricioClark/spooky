''' Definitions of 1- and 2-D periodic grids '''
import numpy as np

class Grid1D:
    def __init__(self, Lx, Nx, dt):
        xx, dx = np.linspace(0, Lx, Nx, endpoint=False, retstep=True)

        ki = np.fft.rfftfreq(Nx, 1/Nx).astype(int)
        kx = 2.0*np.pi*np.fft.rfftfreq(Nx, dx) 
        k2 = kx**2
        kk = np.sqrt(k2)
        
        self.Lx = Lx
        self.xx = xx
        self.dx = dx
        self.dt = dt

        self.kx = kx
        self.k2 = k2
        self.kk = kk
        self.kr = ki

        self.N = Nx
        self.Nx = Nx
        self.shape = (Nx,)

        # Norm and de-aliasing
        self.norm = 1.0/(Nx**2)
        self.zero_mode = 0
        self.dealias_modes = (self.kr > Nx/3)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return np.fft.rfftn(ui)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        return np.fft.irfftn(ui).real

    @staticmethod
    def deriv(ui, ki):
        ''' First derivative in ki direction '''
        return 1.0j*ki*ui

    def avg(self, ui):
        ''' Mean in Fourier space. rfftn is used, so in last dimension middle modes must be 
        doubled to account for negative frequencies. If n is even the last mode contains +fs/2 and -fs/2'''
        tmp = 1 if ui.shape[-1] % 2 == 0 else 2
        sum_ui = np.sum(ui[...,0]) + 2.0*np.sum(ui[...,1:-1]) + tmp* np.sum(ui[...,-1])
        return self.norm * sum_ui

    @staticmethod
    def inner(a, b):
        ''' Inner product '''
        prod = 0.0
        for ca, cb in zip(a, b):
            prod += (ca*cb.conjugate()).real
        return prod

    def energy(self, fields):
        u2  = self.inner(fields, fields)
        eng = 0.5*self.avg(u2)
        return eng

    def enstrophy(self, fields):
        u2  = self.inner(fields, fields)
        ens = 0.5*self.avg(self.k2*u2)
        return ens

    def translate(self, fields, sx):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff*np.exp(1.0j*self.kx*sx) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def deriv_fields(self, fields, ki):
        ''' Compute derivatives in Fourier space in i direction '''
        f = [self.forward(ff) for ff in fields]
        f = [self.deriv(ff, ki) for ff in f]
        return [self.inverse(ff) for ff in f]


class Grid2D(Grid1D):
    def __init__(self, Lx, Ly, Nx, Ny, dt):
        super().__init__(Lx, Nx, dt)

        xi, dx = np.linspace(0, Lx, Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, Ly, Ny, endpoint=False, retstep=True)
        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        kx = 2.0*np.pi*np.fft.fftfreq(Nx, dx) 
        ky = 2.0*np.pi*np.fft.rfftfreq(Ny, dy)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        kk = np.sqrt(k2)

        ki = np.fft.fftfreq(Nx, 1/Nx)
        kj = np.fft.rfftfreq(Ny, 1/Ny)
        ki, kj = np.meshgrid(ki, kj, indexing='ij')
        kr = (ki/Nx)**2 + (kj/Ny)**2

        self.Lx = Lx
        self.Ly = Ly
        self.xx = xx
        self.yy = yy
        self.dy = dy

        self.kx = kx
        self.ky = ky
        self.ki = ki
        self.kj = kj
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.Nx = Nx
        self.Ny = Ny
        self.N  = Nx*Ny
        self.shape = (Nx, Ny)

        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(Nx**2*Ny**2)
        self.zero_mode = (0, 0)
        self.dealias_modes = (kr > 1/9)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.pxx = np.nan_to_num(1.0 - self.kx**2/k2)
            self.pyy = np.nan_to_num(1.0 - self.ky**2/k2)
            self.pxy = np.nan_to_num(- self.kx*self.ky/k2)

    def translate2D(self, fields, sx, sy):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff *np.exp(1.0j*self.kx*sx) *np.exp(1.0j*self.ky*sy) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def inc_proj(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv


class Grid2D_semi(Grid1D):
    ''' 2D grid periodic only in the horizontal direction. To be used with the SPECTER wrapper '''
    def __init__(self, Lx, Lz, Nx, Nz, dt):
        super().__init__(Lx, Nx, dt)
        zi, dz = np.linspace(0, Lz, Nz, endpoint=False, retstep=True)
        kx, zz = np.meshgrid(self.kx, zi, indexing='ij')

        self.Lx = Lx
        self.Lz = Lz
        self.zi = zi
        self.dz = dz
        self.zz = zz
        self.kx = kx

        self.Nx = Nx
        self.Nz = Nz
        self.N  = Nx*Nz
        self.shape = (Nx, Nz)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return np.fft.rfft(ui, axis = 0)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        ui =  np.fft.irfft(ui, axis = 0).real
        return ui


class Grid3D(Grid1D):
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, dt):
        super().__init__(Lx, Nx, dt)

        xi, dx = np.linspace(0, Lx, Nx, endpoint=False, retstep=True)
        yi, dy = np.linspace(0, Ly, Ny, endpoint=False, retstep=True)
        zi, dz = np.linspace(0, Lz, Nz, endpoint=False, retstep=True)
        xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')

        kx = 2.0*np.pi*np.fft.fftfreq(Nx, dx) 
        ky = 2.0*np.pi*np.fft.fftfreq(Ny, dy)
        kz = 2.0*np.pi*np.fft.rfftfreq(Nz, dz)
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        kk = np.sqrt(k2)

        ki = np.fft.fftfreq(Nx, 1/Nx)
        kj = np.fft.fftfreq(Ny, 1/Ny)
        kl = np.fft.rfftfreq(Nz, 1/Nz)
        ki, kj, kl = np.meshgrid(ki, kj, kl, indexing='ij')
        kr = (ki/Nx)**2 + (kj/Ny)**2 + (kz/Nz)**2

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.xx = xx
        self.yy = yy
        self.dy = dy

        self.kx = kx
        self.ky = ky
        self.ki = ki
        self.kj = kj
        self.kz = kz
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N  = Nx*Ny*Nz
        self.shape = (Nx, Ny, Nz)

        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(Nx**2*Ny**2*Nz**2)
        self.zero_mode = (0, 0, 0)
        self.dealias_modes = (self.kr > 1/9)

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     self.pxx = np.nan_to_num(1.0 - self.kx**2/k2)
        #     self.pyy = np.nan_to_num(1.0 - self.ky**2/k2)
        #     self.pzz = np.nan_to_num(1.0 - self.kz**2/k2)
        #     #TODO: check if this is correct
        #     self.pxy = np.nan_to_num(- self.kx*self.ky/k2)
        #     self.pxyz = np.nan_to_num(self.kx*self.ky*self.kz/k2)

    def translate3D(self, fields, sx=0., sy=0., sz=0.):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff *np.exp(1.0j*self.kx*sx) *np.exp(1.0j*self.ky*sy) *np.exp(1.0j*self.kz*sz) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    # def inc_proj(self, fields):
    #     #TODO: fix
    #     ''' Project onto solenoidal modes '''
    #     fu = fields[0]
    #     fv = fields[1]
    #     return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv
