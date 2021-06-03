"""
We compare the tightness of the PAC-Bayes Bennett bound to the PAC-Bayes Bernstein bound.
We consider only the gibbs loss and its variance, not the tandem loss. Thus, the random variables lie in [0,1].
"""

import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
from math import log, sqrt, e, ceil


""" Bernstein Bound:
    Given the empirical gibbs and the variance, return the PAC-Bayes-Bernstein bound
"""
def PBBernstein_bound(emp_gibbs, var_bound, delta, kl, n):
    # compute gam_star
    a = min(var_bound, 1./4)
    b = (kl  + log(1/delta))/n
    gam_star = sqrt(b / ((e-2)*a))
    gam_star = min(gam_star, 1.)
    
    return emp_gibbs + (e-2)*gam_star*a + b/gam_star

""" Bennett Bound:
    Given the empirical gibbs and the variance, return the PAC-Bayes-Bennett bound
"""
def VarCoeff(gam):
    return (e**gam - gam - 1) / gam
    
def PBBen_bound(emp_gibbs, var, delta, kl, n):
    # compute gam_star
    a = min(var, 1./4)
    b = (kl  + log(1/delta))/n
    c = 1/e * (b/a - 1)
    gam_star = 1.+lambertw(c, k=0).real
    
    return emp_gibbs + VarCoeff(gam_star)*a + b/gam_star

# Define the eps term
kl = 5
delta = 0.05
n = 10000

""" construct the grid """
npoints = 200
# Define grid for empirical gibbs and tandem loss
gibbs = np.logspace(-3.4, -0.4, npoints)
var = np.logspace(-3.4, -0.4, npoints)
axis_gibbs, axis_var = np.meshgrid(gibbs, var) # axis


""" true variance """
PBBen = np.zeros((npoints,npoints))
PBBernstein = np.zeros((npoints,npoints))

for i in range(npoints):
    for j in range(npoints):
        # if variance is larger than gibbs
        if (axis_var[i,j]> axis_gibbs[i,j]):
            PBBen[i,j] = np.nan
            PBBernstein[i,j] = np.nan
        else:
            PBBen[i,j] = PBBen_bound(axis_gibbs[i,j], axis_var[i,j], delta, kl, n)
            PBBernstein[i,j] = PBBernstein_bound(axis_gibbs[i,j], axis_var[i,j], delta, kl, n)

""" plot """
def plot():
    fig = plt.figure(figsize=(12,9.6))
    
    plt.rcParams.update({
    'font.size': 30,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    plt.contourf(axis_gibbs, axis_var, PBBen/PBBernstein, levels=25, cmap = "rainbow")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.locator_params(nbins=5)
    plt.title('PB-Bennett/PB-Bernstein', fontsize=45)
    plt.xlabel(r"$\mathbb{E}_\rho[\hat{\tilde{L}}(h,S)] $", fontsize=45)
    plt.ylabel(r"$\mathbb{E}_{\rho}[\tilde\mathbb{V}(h)]$", fontsize=45)
    plt.xticks(fontsize=42, rotation=30)
    plt.yticks(fontsize=42)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('Ben_vs_Bern_'+str(n)+'.png')
plot()