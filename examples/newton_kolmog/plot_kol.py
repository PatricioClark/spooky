import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from pySPEC.solvers import KolmogorovFlow
import params as pm

pm.Lx = 2*np.pi*pm.L
pm.Ly = 2*np.pi*pm.L

# Initialize solver
solver = KolmogorovFlow(pm)

def plot_fields(path, step):
    '''Plot fields'''

    fields = solver.load_fields(path, step)
    spath = f'Imgs/fields'
    os.makedirs(spath, exist_ok = True)
    for ftype, field in zip(solver.ftypes, fields):
        plt.figure()
        plt.imshow(field.T, cmap='viridis')
        plt.title(ftype)
        plt.savefig(f'{spath}/{ftype}.{step}.png', dpi = 300)
        plt.show()

def plot_oz(path, step, convert = True):	
    '''Plot output'''
    if convert:
        fields = solver.load_fields(path, step)
        oz = solver.oz(fields)
    else:
        oz = np.load(f'{path}oz.{step:0{pm.ext}}.npy')

    plt.figure()
    plt.imshow(oz.T, cmap='viridis')
    plt.title('oz')
    plt.savefig(f'{path}oz.{step}.png', dpi = 300)
    plt.show()

def plot_forcing():
    '''Plot forcing'''
    fx = grid.inverse(solver.fx)
    fy = grid.inverse(solver.fy)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(fx.T, cmap='viridis')
    ax[0].set_title('fx')
    ax[1].imshow(fy.T, cmap='viridis')
    ax[1].set_title('fy')
    plt.savefig(f'forcing.png', dpi = 300)
    plt.show()

def GetSpacedElements(list, numElems = 4):
    '''returns evenly spaced elements from list'''
    idxs = np.round(np.linspace(0, len(list)-1, numElems)).astype(int)
    return [list[idx] for idx in idxs] 

def plot_errors(n_newt = None):
    '''plots newton errors and gmres errors'''
    
    fig, axs = plt.subplots(1, 2, figsize=(15,6.5))#, constrained_layout = True)

    font_size = 18

    #plot |b|:
    iter_newt, b_error = np.loadtxt(f'prints/solver.txt', delimiter = ',', skiprows = 1, unpack = True, usecols = (0,1))
    
    #remove repeated (restart) values
    _,idxs = np.unique(iter_newt, return_index=True)
    iter_newt, b_error = iter_newt[idxs], b_error[idxs]

    if n_newt:
        iter_newt = iter_newt[:n_newt]
        b_error = b_error[:n_newt]

    #plot |b| vs newt_it
    ax = axs[0]

    # cmap = plt.get_cmap('viridis')
    alpha = 1
    ax.plot(iter_newt, b_error, color = 'royalblue', alpha = alpha, marker = 'o')

    ax.set_yscale('log')
    ax.set_xlabel(r'$i_N$', fontsize = font_size)
    ax.set_ylabel(r'$\Vert \mathbf{F} \Vert$', fontsize = font_size, labelpad = 8)#TODO: aclaro de alguna forma que es el relativo?

    # Set font size for ticks
    ax.tick_params(axis='both', labelsize=font_size)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #plot gmres data
    ax = axs[1]
    # list_iters = [i for i in range(1, int(max(iter_newt)+1))]
    list_iters = [i for i in range(1, int(max(iter_newt)))]
    # list_iters = GetSpacedElements(list_iters, 6)

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    max_iter = 0
    for i, iter in enumerate(list_iters):
        gmres_iter, error = np.loadtxt(f'prints/error_gmres/iN{iter:02}.txt',\
                            delimiter = ',', skiprows = 1, ndmin = 2, unpack = True)
        # if iter in spaced_iters:
        max_iter = max(max_iter, gmres_iter[-1])
        ax.plot(gmres_iter,error, marker = 'o', color = cmap(i/len(list_iters)),label = fr'$i_N = {iter}$')

    # ax.hlines(10**(-5), gmres_iter[0], max_iter+1, ls = 'dashed', color = 'k')
    ax.set_xlabel(r'$i_G$', fontsize = font_size)
    ax.set_ylabel(r'$\Vert A\delta \mathbf{x} - \mathbf{b} \Vert / \, \Vert \mathbf{b} \Vert$', fontsize = font_size, labelpad = 8)
    # ax.set_ylabel(r'$\Vert A_{i_N}x - b_{i_N} \Vert / \, \Vert b_{i_N} \Vert$', fontsize = font_size)
    # ax.set_ylabel(r'$\frac{\Vert Ax - b \Vert}{\Vert b \Vert}$', fontsize = font_size)
    # ax.set_ylabel(r'$\mathsf{Residual}$', fontsize = font_size)
    ax.set_yscale('log')    

    # Set font size for ticks
    ax.tick_params(axis='both', labelsize=font_size)

    # Custom legend
    # handles, labels = ax.get_legend_handles_labels()

    last_color = (len(list_iters)-1)/len(list_iters)
    #Custon legend de bernardo:
    legend_elements = [
    Line2D([0], [0], color=cmap(0), lw=2, label=r'$i_N = 1$'),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], ls='none', color='k', lw=2, label=r'', marker=r'$\cdot$', ms=5),
    Line2D([0], [0], color=cmap(last_color), lw=2, label=fr'$i_N = {list_iters[-1]}$'),
    ]
    ax.legend(handles=legend_elements, labelspacing=0.001, fontsize=font_size)
    plt.subplots_adjust(wspace=5)
    plt.tight_layout()
    os.makedirs('Imgs', exist_ok = True)
    plt.savefig(f'Imgs/errors.png', dpi = 300)
    plt.close()


def plot_balance(iN=None, bpath = None):
    if bpath:
        bpath = bpath
    elif iN:
        bpath = f'balance/iN{iN:02}/balance.dat'
    else:
        raise ValueError('No path or iN given')

    t, eng, dis, inj = np.loadtxt(bpath,unpack=True)

    fig,axes = plt.subplots(1,3, figsize = (13,5))
    axes = axes.flatten()

    axes[0].plot(t, eng)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$E$")
    axes[0].set_title(f"$|E_f - E_i|={abs(eng[-1]-eng[0]):.2e}$")

    # Enstrophy
    axes[1].plot(t, dis)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("$D$")
    axes[1].set_title(f"$|D_f - D_i|={abs(dis[-1]-dis[0]):.2e}$")

    # Injection
    axes[2].plot(t, inj)
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel("$I$")
    axes[2].set_title(f"$|I_f - I_i|={abs(inj[-1]-inj[0]):.2e}$")

    plt.tight_layout()

    os.makedirs(f'Imgs/balance', exist_ok = True)
    if iN:
        spath = f'Imgs/balance/balance_{iN:02}.png'
    else:
        spath = f'Imgs/balance/balance.png'

    plt.savefig(spath, dpi = 300)
    plt.close()

def plot_orbit(iN):
    os.makedirs('Imgs/output', exist_ok = True)

    extent = [0, pm.Lx, 0, pm.Ly]

    n_plots = 8 #amount of frames to plot
    files = os.listdir(f"output/iN{iN:02}/")
    files = sorted({int(f.split('.')[1]) for f in files}) #Get steps number
    nmbs = GetSpacedElements(files, n_plots)
    
    for i, ftype in enumerate(solver.ftypes):
        fig, axs = plt.subplots(2, n_plots//2, figsize=(20, 10))
        axs = axs.flatten()
        for j, ax in enumerate(axs):
            fields = solver.load_fields(f'output/iN{iN:02}/', nmbs[j])
            data = fields[i]
        
            cax = ax.imshow(data.T, extent=extent, vmin = -2., vmax = 2.)
            t = nmbs[j]*pm.dt
            ax.set_title(f'{type_data}. t={round(t,2)} step={nmbs[j]}')
        # cbar = plt.colorbar(cax, ax = ax)
        cbar = fig.colorbar(cax, ax=axs,shrink=0.5, orientation = 'horizontal')#, pad = .02)

        plt.savefig(f'Imgs/output/{type_data}.{iN:02}.png', dpi = 300)
        plt.close() 

def main():

    plot_errors() #Plots Newton and GMRes errors
    iN = 2 #Newton iteration to plot
    plot_balance(iN) 
    plot_orbit(iN) #Plot 8 frames of fields of orbit

if __name__ == '__main__':
    main()