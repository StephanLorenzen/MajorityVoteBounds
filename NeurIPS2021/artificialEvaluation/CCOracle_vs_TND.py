"""
We compare the CC oracle bounds (Theorem 5, 7, 8) to the TND bound.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib

""" Construct the grid """
npoints = 1000
gibbs = np.linspace(0.005, 0.495, npoints)
tandem = np.linspace(0.005, 0.495, npoints)
grid_gibbs, grid_tandem = np.meshgrid(gibbs, tandem)


""" Calculate the bounds """
## Tandem bound
TND = 4 * grid_tandem

## CCOracle bound
mu_star = (0.5*grid_gibbs - grid_tandem) / (0.5 - grid_gibbs)
CCOracle = (grid_tandem - 2*mu_star*grid_gibbs + mu_star**2) / (0.5 - mu_star)**2

for i in range(npoints):
    for j in range(npoints):
        # Tandem > Gibbs is not possible
        if (grid_tandem[i, j] > grid_gibbs[i, j]):
            mu_star[i,j]=np.nan
            TND[i,j]=np.nan
            CCOracle[i,j]=np.nan

        # Gibbs**2 > Tandem is not possible
        if (grid_gibbs[i, j]**2 > grid_tandem[i, j]):
            mu_star[i,j]=np.nan
            TND[i,j]=np.nan
            CCOracle[i,j]=np.nan

""" Plot the ratio of CCOrale and TND """
def plot():
    fig = plt.figure(figsize=(12,9.6))
    
    plt.rcParams.update({
        'font.size': 30,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    # plot the contourf
    plt.contourf(grid_gibbs, grid_tandem, CCOracle / TND, levels = 30, cmap = "rainbow")
    
    # plot the line when mu=0
    plt.plot(gibbs, 0.5 * gibbs, c='black', linewidth=4, label=r"$\mathbb{E}_{\rho^2}[L(h,h')]=0.5\mathbb{E}_{\rho}[L(h)]$")
    
    # plot the bar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.locator_params(nbins=8)
    
    # plot title and labels
    plt.title('RHS-Thm7 / RHS-Thm3', fontsize=45)
    plt.xlabel(r"$\mathbb{E}_{\rho}[L(h)]$", fontsize=45)
    plt.xticks([0.1, 0.2, 0.3, 0.4], fontsize=42)
    plt.ylabel(r"$\mathbb{E}_{\rho^2}[L(h,h')]$", fontsize=45)
    plt.yticks([0.1, 0.2, 0.3, 0.4], fontsize=42)
    plt.legend(loc='upper left', framealpha=0., handlelength=1, fontsize=40)
    plt.tight_layout()
    plt.savefig('CCOraacle_vs_TND.png')

plot()

