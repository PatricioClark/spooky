'''
Pseudo-spectral solver for the 1D Kuramoto-Sivashinsky equation
'''

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np

import spooky as sp
from spooky.solvers import KuramotoSivashinsky

def load_config(path):
    """Safely load a YAML configuration file."""
    with open(path, "r") as f:
        dic = yaml.safe_load(f)
    return SimpleNamespace(**dic)

def initial_conditions(pm, grid):
    # Initial conditions
    uu = (0.3*np.cos(2*np.pi*3.0*grid.xx/pm.Lx) +
        0.4*np.cos(2*np.pi*5.0*grid.xx/pm.Lx) +
        0.5*np.cos(2*np.pi*4.0*grid.xx/pm.Lx)
        )
    fields = [uu]
    return fields

def plot_fields(uu, title):
    # Plot initial conditions
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(grid.xx, uu)
    ax.set_title('uu')
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
    pm = load_config("params_ks.yaml")   # Solver physics params
    pm.Lx *= 2 * np.pi

    #  Initialize solver
    grid   = sp.Grid1D(pm)
    solver = KuramotoSivashinsky(pm)

    #  Load Initial Conditions
    fields = initial_conditions(pm, grid)

    # Plot initial fields
    uu = fields[0]
    plot_fields(uu, 'initial_fields')

    # Evolve
    fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep, ostep=pm.ostep)

    # Plot Balance
    bal = np.loadtxt('balance.dat', unpack=True)
    plot_balance(bal)

    # Plot final fields
    uu, vv = fields
    plot_fields(uu, vv, 'final_fields')

    # Plot fields
    acc = []
    for ii in range(0,int(pm.T/pm.dt), pm.ostep):
        out = np.load(f'uu.{ii:04}.npy')
        acc.append(out)
    
    acc = np.array(acc)
    plt.figure()
    plt.imshow(acc, extent=[0,pm.Lx,0,pm.T], aspect='auto')
    plt.savefig('acc.png')
    plt.show()


if __name__ == "__main__":
    main()