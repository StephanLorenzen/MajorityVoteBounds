"""
We compare the tightness of the PAC-Bayes-Bennett bound to the PAC-Bayes-Bernstein bound.
We consider only the Gibbs loss and its variance, not the tandem loss.
Thus, the random variable lies in [0,1].
"""

import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
from math import log, sqrt, e, ceil


""" Bernstein Bound:
    Given the empirical loss and the variance, return the PAC-Bayes-Bernstein bound.
    See Theorem 9 in the paper.
"""
def PBBernstein_bound(emp_loss, var, delta, kl, n):

    a = min(var, 1./4) # Variance is bounded by 1/4
    b = (kl  + log(1/delta))/n
    
    # Compute gam_star
    gam_star = sqrt(b / ((e-2)*a))
    gam_star = min(gam_star, 1.) # The range of gam_star is bounded
    
    return emp_loss + (e-2)*gam_star*a + b/gam_star

""" Bennett Bound:
    Given the empirical loss and the variance, return the PAC-Bayes-Bennett bound.
    See Theorem 9 in the paper.
"""

# Calculate the coefficient of variance given gamma
def VarCoeff(gam):
    return (e**gam - gam - 1) / gam
    
def PBBennett_bound(emp_loss, var, delta, kl, n):

    a = min(var, 1./4) # Variance is bounded by 1/4
    b = (kl  + log(1/delta))/n
    
    # Compute gam_star
    c = 1/e * (b/a - 1)
    gam_star = 1.+lambertw(c, k=0).real
    
    return emp_loss + VarCoeff(gam_star)*a + b/gam_star

""" Define the complexity term """
kl = 5
delta = 0.05
n = 10000

""" Construct the grid """
npoints = 200
loss = np.logspace(-3.4, -0.4, npoints)
var = np.logspace(-3.4, -0.4, npoints)
axis_loss, axis_var = np.meshgrid(loss, var) # axis

""" Calculate the bounds """
PBBennett = np.zeros((npoints,npoints))
PBBernstein = np.zeros((npoints,npoints))

for i in range(npoints):
    for j in range(npoints):
        # if variance is larger than loss
        if (axis_var[i,j]> axis_loss[i,j]):
            PBBennett[i,j] = np.nan
            PBBernstein[i,j] = np.nan
        else:
            PBBennett[i,j] = PBBennett_bound(axis_loss[i,j], axis_var[i,j], delta, kl, n)
            PBBernstein[i,j] = PBBernstein_bound(axis_loss[i,j], axis_var[i,j], delta, kl, n)

""" Plot the ratio of PAC-Bayes-Bennett and PAC-Bayes-Bernstein """
def plot():
    fig = plt.figure(figsize=(12,9.6))
    
    plt.rcParams.update({
    'font.size': 30,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    plt.contourf(axis_loss, axis_var, PBBennett/PBBernstein, levels=25, cmap = "rainbow")
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