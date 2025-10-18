''' 1D Adjoint Shallow Water Equations '''

import numpy as np
import os
import matplotlib.pyplot as plt

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps
from .swhd_1d import SWHD_1D


class Adjoint_SWHD_1D(PseudoSpectral):
    ''' 1D Adjoint Shallow Water Equations
        ut_ + u ux_ + (u*u_)x + (h-hb) hx_ = 2(u-um)
        ht_ + u hx_ + g ux_  = 2(h-hm)
    where u,h are physical velocity and height fields,
    u_,h_ the adjoint state fields,
    um,hm the physical field measurements,
    and hb is bottom topography condition.
    '''

    num_fields = 2 # adoint fields u_ , h_
    dim_fields = 1
    def __init__(self, pm, swhd_instance):
        super().__init__(pm)
        self.swhd = swhd_instance
        self.grid = ps.Grid1D(pm)
        self.inverse_u = pm.inverse_u
        self.inverse_u0 = pm.inverse_u0
        self.inverse_h0 = pm.inverse_h0
        self.noise = pm.noise
        self.uum_noise_std = pm.uum_noise_std
        self.hhm_noise_std = pm.hhm_noise_std
        self.iit = pm.iit
        self.iitN = pm.iitN
        self.ckpt = pm.ckpt
        self.Nt = round(pm.T/pm.dt)
        self.total_steps =  int(self.pm.T/self.pm.dt) + 1 # total time steps, since RK runs for Nt+1 steps
        self.step = 0 # current step for adjoint solver
        self.data_path = pm.data_path
        self.field_path = pm.field_path
        self.hb_path = pm.hb_path
        self.hb = None
        self.hbs = None
        self.dg = None
        self.dgs = None
        self.du0 =  None
        self.du0s =  None
        self.dh0 =  None
        self.dh0s =  None
        self.true_hb = None
        self.hx_uu = None
        self.h_ux = None
        self.uus = None
        self.hhs = None
        self.uums = None
        self.hhms = None
        self.uums_sparse = None
        self.hhms_sparse = None
        self.uums_ = None # not none if noise is added to measurements
        self.hhms_ = None # not none if noise is added to measurements
        self.forced_uus = None
        self.forced_hhs = None
        self.u_loss = None
        self.h_loss = None
        self.u0_loss = None
        self.h0_loss = None
        self.val = None
        self.uus_ = None
        self.uu0_ = None
        self.hhs_ = None
        self.hh0_ = None
        self.uu0s = None
        self.uu0 = None
        self.hh0s = None
        self.hh0 = None
        self.true_uu0 = None
        self.true_hh0 = None
        self.st = pm.st
        self.sx = pm.sx
        self.Ns = None
        self.kN = None

    def add_noise(self, field, mean=0.0, std=1.0):
        noise = np.random.normal(loc=mean, scale=std, size=field.shape)
        return field + noise

    def get_measurements(self):
        self.uums = np.load(f'{self.data_path}/uums.npy')[:self.total_steps, :] # all uu fields in time
        self.hhms = np.load(f'{self.data_path}/hhms.npy')[:self.total_steps, :] # all hh fields in time
        # sparsify measurements
        self.uums_sparse, Ns, kN = self.sparsify_mms(self.uums, st = self.st, sx = self.sx)
        self.hhms_sparse, Ns, kN = self.sparsify_mms(self.hhms, st = self.st, sx = self.sx)

        if self.noise:
            self.uums_ = self.uums # to keep pure measurements
            self.hhms_ = self.hhms # to keep pure measurements
            # add noise to measurements
            self.uums = self.add_noise(self.uums, std=self.uum_noise_std)
            self.hhms = self.add_noise(self.hhms, std=self.hhm_noise_std)
            # sparsify measurements
            self.uums_sparse, Ns, kN = self.sparsify_mms(self.uums, st = self.st, sx = self.sx)
            self.hhms_sparse, Ns, kN = self.sparsify_mms(self.hhms, st = self.st, sx = self.sx)
            np.save(f'{self.hb_path}/uums_noise', self.uums)
            np.save(f'{self.hb_path}/hhms_noise', self.hhms)

    def sparsify_mms(self, field, st = 1, sx = 1, N = 1024):
            '''Returns sparse signal, number of sparse measurements and Nyquist frequency'''

            # Create a copy to avoid modifying the original
            modified_data_ = np.zeros_like(field)

            # Identify indices to keep based on the time interval
            time_indices = np.arange(0, field.shape[0], st)
            # Apply the time-interval-based filter
            modified_data_[time_indices] = field[time_indices]

            # Create another copy to avoid modifying the original
            modified_data = np.zeros_like(modified_data_)
            # Identify indices to keep based on the space interval
            space_indices = np.arange(0, field.shape[-1], sx)
            modified_data[:,space_indices] = modified_data_[:,space_indices]
            # calculate effective grid size Ns and Nyquist frequency
            Ns = int(N/sx)
            kN = int(Ns/2)

            return modified_data, Ns, kN

    def get_sparse_forcing(self):
        if self.inverse_u:
            self.forced_uus = np.zeros_like(self.uums)
        else:
            self.forced_uus, Ns, kN = self.sparsify_mms(self.uus - self.uums, st = self.st, sx = self.sx)
        self.forced_hhs, self.Ns, self.kN = self.sparsify_mms(self.hhs - self.hhms, st = self.st, sx = self.sx)

    def update_true_hb(self):
        self.true_hb = np.load(f'{self.data_path}/hb.npy')

    def update_fields(self, swhd_instance):
        # update the forward solver status first
        self.swhd = swhd_instance
        self.uus = self.swhd.uus # all uu fields in time
        self.hhs = self.swhd.hhs # all hh fields in time

    # def sparsify(self):
    #     self.uuforcings = self.uus -self.uums[:self.Nt,:]
    #     self.hhforcings = self.hhs -self.hhms[:self.Nt,:]

    def update_loss(self, iit):
        if self.noise:
            u_loss = np.sum((self.uums_ - self.uus)**2)
            h_loss = np.sum((self.hhms_ - self.hhs)**2)

        else:
            u_loss = np.sum((self.uums - self.uus)**2)
            h_loss = np.sum((self.hhms - self.hhs)**2)
        if self.inverse_u0:
            u0_loss = np.sum((self.uums[0] - self.uus[0])**2)
            self.u0_loss = self.save_to_ram(self.u0_loss, u0_loss, iit, self.iitN, dtype=np.float64)

        if self.inverse_h0:
            h0_loss = np.sum((self.hhms[0] - self.hhs[0])**2)
            self.h0_loss = self.save_to_ram(self.h0_loss, h0_loss, iit, self.iitN, dtype=np.float64)
        self.u_loss = self.save_to_ram(self.u_loss, u_loss, iit, self.iitN, dtype=np.float64)
        self.h_loss = self.save_to_ram(self.h_loss, h_loss, iit, self.iitN, dtype=np.float64)


    def update_val(self, iit):
        val = np.sum((self.true_hb - self.hb)**2)
        self.val = self.save_to_ram(self.val, val, iit, self.iitN, dtype=np.float64)

    def update_hb(self, hb):
        self.hb = hb

    def update_hbs(self, iit):
        self.hbs = self.save_to_ram(self.hbs, self.hb, iit, self.iitN, dtype=np.float64)


    def update_uu0(self, uu0):
        self.uu0 = uu0

    def update_true_uu0(self, true_uu0):
        self.true_uu0 = true_uu0

    def update_uu0s(self, iit):
        self.uu0s = self.save_to_ram(self.uu0s, self.uu0, iit, self.iitN, dtype=np.float64) # save to uu0 history

    def update_hh0(self, hh0):
        self.hh0 = hh0

    def update_true_hh0(self, true_hh0):
        self.true_hh0 = true_hh0

    def update_hh0s(self, iit):
        self.hh0s = self.save_to_ram(self.hh0s, self.hh0, iit, self.iitN, dtype=np.float64) # save to hh0 history

    def update_dg(self, dg):
        self.dg = dg

    def update_dgs(self, iit):
        self.dgs = self.save_to_ram(self.dgs, self.dg, iit, self.iitN, dtype=np.float64)

    def update_du0(self, du0):
        self.du0 = du0

    def update_du0s(self, iit):
        self.du0s = self.save_to_ram(self.du0s, self.du0, iit, self.iitN, dtype=np.float64)

    def update_dh0(self, dh0):
        self.dh0 = dh0

    def update_dh0s(self, iit):
        self.dh0s = self.save_to_ram(self.dh0s, self.dh0, iit, self.iitN, dtype=np.float64)


    def sparsify(self, signal, s=1, N=1024):
        '''Masks signal to make sparse measurements every s points.
        Returns sparse signal, Nyquist frequency and number of sparse measurements'''

        Ns = int(N/s)
        kN = int(Ns/2)
        signal_sparse = np.zeros_like(signal)
        signal_sparse[::s] = signal[::s]

        return signal_sparse,kN,Ns

    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fu_  = fields[0]
        fu_p = prev[0]

        fh_  = fields[1]
        fh_p = prev[1]

        # Access the step from the evolve method via the class attribute
        step = self.step
        # get physical fields and measurements from T to t0, back stepping in time
        Nt = self.Nt
        back_step = Nt-1 - step
        uu = self.uus[back_step]
        hh = self.hhs[back_step]
        # get hb
        hb = self.hb
        # forcing terms
        forced_uu = self.forced_uus[back_step]
        forced_hh = self.forced_hhs[back_step]

        fu  = self.grid.forward(uu)
        fux = self.grid.deriv(fu, self.grid.kx)
        ux = self.grid.inverse(fux)

        # fh  = self.grid.forward(hh)

        # calculate terms
        uu_ = self.grid.inverse(fu_)
        # hh_ = self.grid.inverse(fh_)

        fux_ = self.grid.deriv(fu_, self.grid.kx)
        ux_ = self.grid.inverse(fux_)
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)

        fu_ux_ = self.grid.forward(uu*ux_)
        fu_u_x = self.grid.deriv(self.grid.forward(uu*uu_), self.grid.kx)
        fu_hx_ = self.grid.forward(uu*hx_)

        fh_hb_hx_ = self.grid.forward((hh-hb)*hx_)

        # Fourier transform sparse forced terms
        fforced_uu = self.grid.forward(forced_uu)
        fforced_hh = self.grid.forward(forced_hh)
        # get rid of frequencies larger than Nyquist for subsampled grid
        fforced_uu[self.kN+1:] = 0.0
        fforced_hh[self.kN+1:] = 0.0
        # normalize in Fourier space
        fforced_uu = fforced_uu*1024/self.Ns
        fforced_hh = fforced_hh*1024/self.Ns
        # backwards integration in time, with sampled forcing terms
        fu_ = fu_p - (self.grid.dt/oo) * (2*fforced_uu - fu_ux_ - fu_u_x - fh_hb_hx_)
        fh_ = fh_p - (self.grid.dt/oo) * (2*fforced_hh - self.pm.g*fux_ - fu_hx_)

        # de-aliasing
        # fu_[self.grid.zero_mode] = 0.0
        fu_[self.grid.dealias_modes] = 0.0
        fh_[self.grid.zero_mode] = 0.0
        fh_[self.grid.dealias_modes] = 0.0


        # GD step
        # update new hx_
        fhx_ = self.grid.deriv(fh_, self.grid.kx)
        hx_ = self.grid.inverse(fhx_)
        # multiply with field
        non_dealiased_dg_ = hx_* uu
        fdg_ = self.grid.forward(non_dealiased_dg_)
        # de-aliasing GD step
        fdg_[self.grid.dealias_modes] = 0.0
        dg_ = self.grid.inverse(fdg_)
        self.hx_uu = self.save_to_ram(self.hx_uu, dg_, step, self.total_steps-1, dtype=np.float64) # starts saving at step 1, not 0
        return [fu_,fh_]


    def save_to_ram(self, storage, new_data, step, total_steps, dtype=np.float64):
        """
        Saves new data to an in-memory storage array.

        Args:
            storage (np.ndarray or None): The in-memory storage array. Pass `None` on the first call to initialize it.
            new_data (np.ndarray): Data to be saved at the current iteration.
            step (int): Current iteration (used to index the storage array).
            total_steps (int): Total number of iterations to preallocate space for.
            dtype (type): Data type of the saved array (default: np.float64).

        Returns:
            np.ndarray: Updated storage array.
        """
        if step == 0:
            # Initialize the in-memory storage array with preallocated space
            # storage = np.zeros((total_steps,) + new_data.shape, dtype=dtype)
            storage = np.full((total_steps,) + new_data.shape, np.nan, dtype=dtype)


        # Save the new data into the corresponding slot
        storage[step] = new_data

        return storage
# Save hb and dg arays in file
    def save_memmap(self, filename, new_data, step, total_steps, dtype=np.float64):
        """
        Saves new data to an existing or new preallocated memory-mapped .npy file.

        Args:
            filename (str): Path to the memory-mapped .npy file.
            new_data (np.ndarray): Data to be saved at the current iteration.
            iit (int): Current iteration (used to index the memory-mapped file).
            total_iterations (int): Total number of iterations to preallocate space for.
            dtype (type): Data type of the saved array (default: np.float64).
        """
        if step == 0:
            # Create a new memory-mapped file with preallocated space for all iterations
            if os.path.exists(filename):
                os.remove(filename)
            # Shape includes space for total_iterations along the first axis
            shape = (total_steps,) + new_data.shape  # Preallocate for total iterations
            fp = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
        else:
            # Load the existing memory-mapped file (no need to resize anymore)
            fp = np.load(filename, mmap_mode='r+')

        # Write new data into the current iteration slot
        fp[step] = new_data
        del fp  # Force the file to flush and close

    def outs(self, fields, step, opath):
        # save current step for adjoint solver
        self.step = step
        uu_ = self.grid.inverse(fields[0])
        self.uus_ = self.save_to_ram(self.uus_, uu_, int(step/self.pm.ostep), int(self.total_steps/self.pm.ostep), dtype=np.float64)

        hh_ = self.grid.inverse(fields[1])
        self.hhs_ = self.save_to_ram(self.hhs_, hh_, int(step/self.pm.ostep), int(self.total_steps/self.pm.ostep), dtype=np.float64)


    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
