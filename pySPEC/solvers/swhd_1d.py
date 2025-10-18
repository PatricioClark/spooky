''' 1D Shallow Water Equations '''

import numpy as np
import os

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps


class SWHD_1D(PseudoSpectral):
    ''' 1D Shallow Water Equations
        ut + u ux + g hx = 0
        ht + ( u(h-hb) )x = 0
    where u,h are velocity and height fields,
    and hb is bottom topography condition.
    '''

    num_fields = 2
    dim_fields = 1
    def __init__(self, pm):
        super().__init__(pm)
        self.grid = ps.Grid1D(pm)
        self.make_data = pm.make_data
        self.noise = pm.noise
        self.uum_noise_std = pm.uum_noise_std
        self.hhm_noise_std = pm.hhm_noise_std
        self.iit = pm.iit
        self.iit0 = pm.iit0
        self.iitN = pm.iitN
        self.total_iterations = self.iitN - self.iit0 - 1
        self.total_steps =  int(self.pm.T/self.pm.dt)+1 # total time steps, since RK runs for Nt+1 steps
        self.step = 0 # current step for adjoint solver
        self.data_path = pm.data_path
        self.hb_path = pm.hb_path
        self.uus = None
        self.hhs = None
        self.uus_noise = None
        self.hhs_noise = None
        self.hb = None
        self.true_hb = None

    def update_true_hb(self):
        self.true_hb = np.load(f'{self.data_path}/hb.npy')

    def update_hb(self, hb):
        self.hb = hb

    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        fh  = fields[1]
        fhp = prev[1]

        hb = self.hb

        # Non-linear term
        uu = self.grid.inverse(fu)
        hh = self.grid.inverse(fh)

        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))

        fhx = self.grid.deriv(fh, self.grid.kx) # i k_i fh_i

        fu_ux = self.grid.forward(uu*ux)
        fu_h_hb_x = self.grid.deriv(self.grid.forward(uu*(hh-hb)) , self.grid.kx) # i k_i f(uu*(hh-hb))_i

        fu = fup + (self.grid.dt/oo) * (-fu_ux -  self.pm.g*fhx)
        fh = fhp + (self.grid.dt/oo) * (-fu_h_hb_x)

        # de-aliasing
        fu[self.grid.zero_mode] = 0.0
        fu[self.grid.dealias_modes] = 0.0
        # fh[self.grid.zero_mode] = 0.0 # zero mode for h is not null
        fh[self.grid.dealias_modes] = 0.0

        return [fu, fh]

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
            storage = np.zeros((total_steps,) + new_data.shape, dtype=dtype)
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

    def add_noise(self, field, mean=0.0, std=1.0):
        noise = np.random.normal(loc=mean, scale=std, size=field.shape)
        return field + noise

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        self.uus = self.save_to_ram(self.uus, uu, int(step/self.pm.ostep), int(self.total_steps/self.pm.ostep), dtype=np.float64)

        hh = self.grid.inverse(fields[1])
        self.hhs = self.save_to_ram(self.hhs, hh, int(step/self.pm.ostep), int(self.total_steps/self.pm.ostep), dtype=np.float64)
        if self.make_data and (step == self.total_steps-1):

            np.save(f'{self.pm.out_path}/uums', self.uus)
            np.save(f'{self.pm.out_path}/hhms', self.hhs)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)
