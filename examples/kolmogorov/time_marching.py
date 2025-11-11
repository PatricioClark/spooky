'''
Pseudo-spectral solver for the 1- and 2-D periodic PDEs

A variable-order RK scheme is used for time integrationa
and the 2/3 rule is used for dealiasing.
'''

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np

import spooky as sp
from spooky.solvers import KolmogorovFlow

def load_config(path):
    """Safely load a YAML configuration file."""
    with open(path, "r") as f:
        dic = yaml.safe_load(f)
    return SimpleNamespace(**dic)

def initial_conditions(pm, grid):
    # Initial conditions
    uu = np.cos(2*np.pi*1.0*grid.yy/pm.Lx) + 0.1*np.sin(2*np.pi*2.0*grid.yy/pm.Lx)
    vv = np.cos(2*np.pi*1.0*grid.xx/pm.Lx) + 0.2*np.cos(3*np.pi*2.0*grid.yy/pm.Lx)
    fu = grid.forward(uu)
    fv = grid.forward(vv)
    fu, fv = grid.inc_proj([fu, fv])
    uu = grid.inverse(fu)
    vv = grid.inverse(fv)
    fields = [uu, vv]
    return fields, uu, vv

def plot_fields(uu, vv, title):
    # Plot initial conditions
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(uu.T, cmap='viridis')
    ax[0].set_title('uu')
    ax[1].imshow(vv.T, cmap='viridis')
    ax[1].set_title('vv')
    plt.savefig(f'{title}.png', dpi = 300)
    plt.show()
    plt.close()

def plot_balance(balance):
    plt.figure()
    plt.plot(balance[0], balance[1])
    plt.savefig('balance.png', dpi = 300)
    plt.show()
    plt.close()    

def main():
    #  Load configs 
    pm = load_config("params_kolmog.yaml")   # Solver physics params
    pm.Lx *= 2 * np.pi
    pm.Ly *= 2 * np.pi

    #  Initialize solver
    grid   = sp.Grid2D(pm)
    solver = KolmogorovFlow(pm)

    #  Load Initial Conditions
    fields, uu, vv = initial_conditions(pm, grid)

    # Plot initial fields
    plot_fields(uu, vv, 'initial_fields')

    # Evolve
    fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep)

    # Plot Balance
    bal = np.loadtxt('balance.dat', unpack=True)
    plot_balance(bal)

    # Plot final fields
    uu, vv = fields
    plot_fields(uu, vv, 'final_fields')

if __name__ == "__main__":
    main()
