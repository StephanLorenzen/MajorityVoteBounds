"""
We compare the PB empirical Bennett bound to the PB-kl bound.
We consider only the gibbs loss and its variance, not the tandem loss. Thus, the random variables lie in [0,1].
"""

import numpy as np
from scipy import optimize
from scipy.special import lambertw
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from math import log, sqrt, e, ceil

# helper functions for kl-1
def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([ q*log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])

def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])
    

def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    #print(q, right_hand_side)
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0-1e-9) <= 0.0:
        return 1.0-1e-9
    else:
        return optimize.brentq(f, q, 1.0-1e-11)


def solve_kl_inf(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x < q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1e-9) <= 0.0:
        return 1e-9
    else:
        return optimize.brentq(f, 1e-11, q)

def Lambert(c, branch):
    """
    Implement the Lambert W function, i.e., the solution of xe^x=c, by Newton's method.
    Requirement : c\geq -1/e
    branch W0 : the solution x\geq -1
    branch W-1 : the solution x\leq -1
    """
    eps = 1e-9
    
    # fx(x, c) = xe^x-c
    def _fx(x, c):
        return x*np.exp(x)-c
        
    # fx_prime(x) = (x+1)e^x, first derivative of _fx
    def _fx_prime(x):
        return (x+1)*np.exp(x)
    
    if branch == 'W0':
        x_old = 0 # initial guess for the principle branch
    elif branch == 'W-1':
        x_old = -5  # initial guess for the -1 branch
    else:
        Warning('No such branch')
        
    diff = abs(_fx(x_old, c))
    while(diff > eps):
        x_new = x_old - _fx(x_old, c)/_fx_prime(x_old)
        x_old = x_new
        diff = _fx(x_old, c)
    
    return x_new


""" PBkl Bound : 
    Given the empirical variance, return the PBkl bound
"""
def PBkl_bound(emp_gibbs):
    rhs1 = (kl+log(2.*sqrt(n)/delta))/n
    return emp_gibbs + solve_kl_sup(emp_gibbs, rhs1) - emp_gibbs


""" Variance Bound :
    Given the empirical variance, return the bound for the expected variance
"""
def variance_bound(emp_var, method='self-bound'):
    delta1 = delta/2.
    
    if method != 'self-bound':
        """ method 1, by kl-1 """
        rhs = (kl+log(2.*sqrt(n)/delta1))/n
        return solve_kl_sup(emp_var, rhs)
    else:
        """ method 2, by self-bounding technique """
        # number of points
        nu1  = 0.5 * sqrt( (n-1)/log(1/delta1)+1 ) + 0.5
        nu1 = ceil(log(nu1)/log(c1))
        # construct the grid
        base = 2*(n-1)/n * 1./(sqrt( (n-1)/log(1/delta1)+1 ) + 1)
        lam_grid = np.array([c1**i * base for i in range(nu1)])
        
        # compute lam_star
        a = emp_var
        b = (kl+log(nu1/delta1))/(2*(n-1))
        t_star = 1./ (sqrt(a/(b)+1)+1 )
        lam_star = 2*(n-1)*t_star/n
        
        # find the closest lam_star in the grid to calculate the bound
        lam_star = lam_grid[np.argmin(abs(lam_grid-lam_star))]
        t_star = lam_star/2. * n/(n-1)
        return (a / (1 - t_star) + b / (t_star * (1 - t_star)), nu1)
    
""" Bernstein Bound:
    Given the empirical gibbs and the bound for the empirical variance, return PBEBernstein_bound
"""
def PBEBernstein_bound(emp_gibbs, var_bound, delta, kl, n):
    """
    #nu1 = var_bound[1]
    
    # number of points
    nu2 = sqrt((e-2)*n / (4*log(1/delta2)))
    nu2 = ceil(log(nu2)/log(c2))
    
    # construct the grid
    base = sqrt(4*log(1/delta2)/((e-2)*n))
    gam_grid = np.array([c2**i * base for i in range(nu2)])
    """
    # compute gam_star
    a = min(var_bound, 1./4)
    b = (kl  + log(1/delta))/n
    gam_star = sqrt(b / ((e-2)*a))
    gam_star = np.min([gam_star, 1])
    
    # find the closest gam_star in the grid to calculate the bound
    #gam_star = gam_grid[np.argmin(abs(gam_grid-gam_star))]
    #print('bernstein gam_star', gam_star)
    return emp_gibbs + (e-2)*gam_star*a + b/gam_star

""" Bennett Bound:
    Given the empirical gibbs and the bound for the empirical variance, return PBEBen_bound
"""
def VarCoeff(gam):
    return (e**gam - gam - 1) / gam
    
def PBEBen_bound(emp_gibbs, var_bound, delta, kl, n):
    """
    # compute gamma_min
    c_min = 1/e * (4/n*log(1/delta) - 1)     
    gam_min = 1. + Lambert(c_min, 'W0')
    """
    """
    # compute gamma_max
    
    nu1 = var_bound[1]
    #nu1  = 0.5 * sqrt( (n-1)/log(1/delta1)+1 ) + 0.5
    #nu1 = ceil(log(nu1)/log(c1))

    alpha = 1./ (1+(n-1)/(2*c1*log(nu1/delta1)))
    c_max = -alpha * e**(-alpha)
    gam_max = - (Lambert(c_max, 'W-1')+alpha)
    
    # computer nu2
    nu2 = ceil(log(gam_max/gam_min)/log(c2))
    """
    """
    # construct the grid
    base = gam_min
    gam_grid = np.array([c2**i * base for i in range(nu2)])
    """
    # compute gam_star
    a = min(var_bound, 1./4)
    b = (kl  + log(1/delta))/n
    c = 1/e * (b/a - 1)
    gam_star = 1.+Lambert(c, 'W0')
    
    # find the closest gam_star in the grid to calculate the bound
    #gam_star = gam_grid[np.argmin(abs(gam_grid-gam_star))]
    #print('bennett gam_star', gam_star)
    return emp_gibbs + VarCoeff(gam_star)*a + b/gam_star

# Define the eps term
kl = 5
delta = 0.05
n = 1000

""" construct the grid """
npoints = 200
# Define grid for empirical gibbs and tandem loss
gibbs = np.logspace(-3.4, -0.4, npoints)
var = np.logspace(-3.4, -0.4, npoints)
axis_gibbs, axis_var = np.meshgrid(gibbs, var) # axis


""" true variance """
PBTBen = np.zeros((npoints,npoints))
PBTBernstein = np.zeros((npoints,npoints))

for i in range(npoints):
    for j in range(npoints):
        # if variance is larger than gibbs
        if (axis_var[i,j]> axis_gibbs[i,j]):
            #PBkl[i,j] = np.nan
            PBTBen[i,j] = np.nan
            PBTBernstein[i,j] = np.nan
        else:
            #PBkl[i,j] = PBkl_bound(axis_gibbs[i,j])
            #PBEBen[i,j] = PBEBen_bound(axis_gibbs[i,j], variance_bound(axis_var[i,j]))
            #PBEBernstein[i,j] = PBEBernstein_bound(axis_gibbs[i,j], variance_bound(axis_var[i,j]))

            PBTBen[i,j] = PBEBen_bound(axis_gibbs[i,j], axis_var[i,j], delta, kl, n)
            PBTBernstein[i,j] = PBEBernstein_bound(axis_gibbs[i,j], axis_var[i,j], delta, kl, n)

""" plot """
def plot1():

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
    ## ax1
    p1 = ax1.contourf(axis_gibbs, axis_var, PBTBernstein/PBkl, levels=20, cmap = "rainbow")
    p3 = ax1.contour(axis_gibbs, axis_var, PBTBernstein/PBkl, [0.6,0.8,1.], colors='k', linestyles='dashed' )
    ax1.clabel(p3, fontsize=10, inline=False)
    ax1.set_title('PB-EBernstein/PB-kl')
    ax1.set_xlabel('Empirical loss')
    #ax1.set_ylabel('Empirical variance')
    ax1.set_ylabel('Expected variance')
    ax1.set_xscale("log")
    ax1.set_yscale("log")


    ## ax2
    p2 = ax2.contourf(axis_gibbs, axis_var, PBTBen/PBkl, levels=20, cmap = "rainbow")
    p4 = ax2.contour(axis_gibbs, axis_var, PBTBen/PBkl, [0.6,0.8,1.], colors='k', linestyles='dashed' )
    ax2.clabel(p4, fontsize=10, inline=False)
    ax2.set_title('PB-EBennett/PB-kl')
    ax2.set_xlabel('Empirical loss')
    #ax2.set_ylabel('Empirical variance')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plt.colorbar(p2, ax=(ax1,ax2))
    plt.savefig('PB-EBennett_vs_PB-kl.png')
#plot1()

def plot2():
    fig = plt.figure()
    plt.contourf(axis_gibbs, axis_var, PBTBen/PBTBernstein, levels=30, cmap = "rainbow")
    #plt.plot(gibbs, [(kl  + log(1/delta))/n/(e-2) for i in range(npoints)], color='k', label='$\gamma^*=1$')
    plt.colorbar()
    #CS = plt.contour(axis_gibbs, axis_var, PBEBen/PBEBernstein, [0.88,0.97], colors='k', linestyles='dashed' )
    #plt.clabel(CS, fontsize=10, inline=True)
    plt.title('PB-Bennett/PB-Bernstein')
    plt.xlabel('Empirical loss')
    #plt.ylabel('Empirical variance')
    plt.ylabel('Expected variance')
    plt.xscale("log")
    plt.yscale("log")
    #plt.legend()
    plt.savefig('PB-Bennett_vs_PB-Berinstein'+str(n)+'.png')
plot2()

def plot3():
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
    ## ax1
    p1 = ax1.contourf(axis_gibbs, axis_var, gamBernstein, cmap = "rainbow")
    #p3 = ax1.contour(axis_gibbs, axis_var, PBTBernstein/PBkl, [0.6,0.8,1.], colors='k', linestyles='dashed' )
    #ax1.clabel(p3, fontsize=10, inline=False)
    ax1.set_title('PB-EBernstein/PB-kl')
    ax1.set_xlabel('Empirical loss')
    #ax1.set_ylabel('Empirical variance')
    ax1.set_ylabel('Expected variance')
    ax1.set_xscale("log")
    plt.colorbar(p1, ax=ax1)
    ax1.set_yscale("log")


    ## ax2
    p2 = ax2.contourf(axis_gibbs, axis_var, gamBen, cmap = "rainbow")
    #p4 = ax2.contour(axis_gibbs, axis_var, PBTBen/PBkl, [0.6,0.8,1.], colors='k', linestyles='dashed' )
    #ax2.clabel(p4, fontsize=10, inline=False)
    ax2.set_title('PB-EBennett/PB-kl')
    ax2.set_xlabel('Empirical loss')
    #ax2.set_ylabel('Empirical variance')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plt.colorbar(p2, ax=ax2)
    plt.savefig('gam_PB-EBennett_vs_PB-EBerinstein.png')
#plot3()