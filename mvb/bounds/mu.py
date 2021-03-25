#
# Implements the MU bound.
#
import numpy as np
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 

# Implementation of MU
def MU(tandem_risk, gibbs_risk, n, n2, KL, mu_grid=[0.0], delta=0.05):
    if gibbs_risk > 0.5:
        return (1.0, mu_grid, 1.0)
    if len(mu_grid)<1:
        return (1.0, mu_grid, 1.0)

    # Union bound over K = len(mu_grid) -> delta = delta/K
    delta /= len(mu_grid)

    # UpperBound_TandemRisk by inverse kl
    rhs_tr = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
    ub_tr  = solve_kl_sup(tandem_risk, rhs_tr)
    
    # LowerBound_GibbsRisk by inverse kl
    rhs_gr = ( KL + log(4.0*sqrt(n)/delta) ) / n
    lb_gr  = solve_kl_inf(gibbs_risk, rhs_gr)
   
    # Compute K bounds
    opt_bnd, opt_muTandemUB, opt_mu = 2000.0, 2000.0, 0.0 # (bound, muTandemUB, mu)
    for mu in mu_grid:
        muTandemUB = ub_tr - 2*mu*lb_gr + mu**2
        bnd = muTandemUB / (0.5-mu)**2
        if bnd < opt_bnd:
            opt_bnd, opt_muTandemUB, opt_mu = bnd, muTandemUB, mu
        elif  bnd > opt_bnd:
        #    # if stop improving, break
            break

    return (min(1.0, opt_bnd), [opt_mu], min(1.0, opt_muTandemUB))

# Optimize MU
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeMU(tandem_risks, gibbs_risks, n, n2, delta=0.05, options=None):
    options = dict() if options is None else options
    if "mu_grid" not in options:
        return _optimizeMU(tandem_risks, gibbs_risks, n, n2, delta=delta, options=None)
    else:
        mu_grid = options['mu_grid']
        delta /= len(mu_grid)
        best_bound = (2,)
        for mu in mu_grid:
            b = _optimizeMU(tandem_risks,gibbs_risks,n,n2,mu=mu,delta=delta,options=None)
            if b[0] < best_bound[0]:
                best_bound = b
            elif b[0] > best_bound[0]:
        #        # if stop improving, break
                break
        return best_bound
        

# Same as above, but mu is now a single value. If None, mu will be optimized
def _optimizeMU(tandem_risks, gibbs_risks, n, n2, mu=None, delta=0.05, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    mu_input = mu

    if optimizer not in ['GD', 'RProp', 'iRProp']:
        warn('optimizeMU: unknown optimizer: \''+optimizer+'\', using iRProp')
        optimizer = 'iRProp'
    
    m = gibbs_risks.shape[0]
    pi  = uniform_distribution(m)
    rho = uniform_distribution(m) 

    def _gr(rho): # Compute Gibbs risk from gibbs_risk list and rho
        return np.average(gibbs_risks, weights=rho)
    def _tnd(rho): # Compute disagreement from disagreements matrix and rho
        return np.average(np.average(tandem_risks, weights=rho, axis=0), weights=rho)
    def _bound(rho, mu=None, lam=None, gam=None): # Compute bound
        rho = softmax(rho)
        gr  = _gr(rho)
        tnd = _tnd(rho)
        KL  = kl(rho,pi)
        if lam is None:
            lam = 2.0 / (sqrt((2.0*n2*tnd)/(2*KL+log(4.0*sqrt(n2)/delta)) + 1) + 1)
        if gam is None:
            gam = min(2.0, sqrt( (2.0*KL+log(16.0*n/delta**2)) / (n*gr) ))

        # Compute upper and lower bounds given lam and gam
        ub_tnd = tnd/(1-lam/2)+(2*KL+log(4*sqrt(n2)/delta))/(lam*(1-lam/2)*n2)
        lb_gr  = (1-gam/2.0)*gr-(KL+log(4*sqrt(n)/delta))/(gam*n)
        
        if mu is None:
            # Compute mu based on our estimates
            mu = (0.5*lb_gr-ub_tnd)/(0.5-lb_gr)
         
        bound  = (ub_tnd - 2*mu*lb_gr + mu**2) / (0.5-mu)**2
        return (bound, mu, lam, gam)

    def _gradient(rho, mu, lam, gam):
        a = 1.0/(1.0-lam/2.0)
        b = mu*(1-gam/2.0)
        c = 1.0/(lam*(1.0-lam/2.0)*n2) + mu/(gam*n)
            
        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho,1.0-Srho))

        return 2*np.dot(a*np.dot(tandem_risks, Srho)-b*gibbs_risks+c*(1+np.log(Srho/pi)), Smat)
        
    max_iterations = options.get('max_iterations', None)
    eps = options.get('eps', 10**-9)
        
    if optimizer=='GD':
        def _optRho(rho, mu, lam, gam):
            lr = options.get('learning_rate', 1)
            return GD(lambda x: _gradient(x,mu,lam,gam), lambda x: _bound(x,mu,lam,gam)[0], rho,\
                    max_iterations=max_iterations, lr=lr, eps=eps)
    elif optimizer=='RProp':
        max_iterations = options.get('max_iterations', 1000)
        def _optRho(rho, mu, lam, gam):
            return RProp(lambda x: _gradient(x,mu,lam,gam),rho,eps=eps,max_iterations=max_iterations)
    elif optimizer=='iRProp':
        max_iterations = options.get('max_iterations', 100)
        def _optRho(rho, mu, lam, gam):
            return iRProp(lambda x: _gradient(x,mu,lam,gam), lambda x: _bound(x,mu,lam,gam)[0], rho,\
                    eps=eps,max_iterations=max_iterations)

    rho = uniform_distribution(m)
    # If mu_input is None, mu will be computed, otherwise mu_input will be returned
    b, mu, lam, gam  = _bound(rho, mu=mu_input)
    bp = b+1
    while abs(b-bp) > eps:
        bp = b
        # Optimize rho
        nrho = _optRho(rho,mu,lam,gam)
        # Optimize lam + gam
        b, nmu, nlam, ngam = _bound(nrho, mu=mu_input)
        if b > bp:
            b = bp
            break
        rho, mu, lam, gam = nrho, nmu, nlam, ngam
    return (min(1.0,b), softmax(rho), mu, lam, gam)

