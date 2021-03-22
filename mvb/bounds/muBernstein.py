#
# Implements the MU Bernstein bound.
#
import numpy as np
from math import log, sqrt, exp, e, ceil, nan
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp


# Implementation of MUBernstein
#def MUBernstein(mutandem_risk, vartandem_risk, n2, KL, mu=0.0, delta=0.05):
def MUBernstein(MVBounds, data, incl_oob, KL, mu_grid=[0.0], delta=0.05):
    best_bound = (2000.0, )
    for mu in mu_grid:
        # Compute the quantities depend on mu
        mutandem_risk, vartandem_risk, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)

        # Compute a bound over the variance from Corollary 17
        varUB, _ = _varMUBernstein(vartandem_risk, n2, KL, mu, delta1= delta/2.)

        # Plug the bound over variance to compute a bound over the muTandem loss following Corollary 20.
        bernTandemUB, _ = _muBernstein(mutandem_risk, varUB, n2, KL, mu, delta2= delta/2.)
  
        # Compute the overall bound
        bnd = bernTandemUB / (0.5-mu)**2
        if bnd < best_bound[0]:
            best_bound = (bnd, [mu], mutandem_risk, vartandem_risk, varUB, bernTandemUB)
        elif bnd > best_bound[0]:
            # if stop improving, break
            break
    return best_bound


#Corollary 20 : \label{cor:pac-bayes-bernstein_grid}
def _muBernstein(mutandem_risk, varMuBound, n2, KL, mu=0.0, c2=1.0, delta2=0.05, unionBound = False):

    if unionBound:
        nu2 = ceil( log( sqrt( (e-2)*n2 / (4*log(1/delta2)) ) ) / log(c2) )
    else:
        nu2 = 1
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)
    
    # E[varMuBound]<=Kmu^2/4
    varMuBound = min(varMuBound, Kmu**2/4)
    
    # From the proof of Collorary 20.
    bprime=c2*(2*KL + log(nu2) -log(delta2))/n2
    a=(e-2)*varMuBound
    gammastar = sqrt(bprime / a)
    
    # The range of Gamma^*
    gam_lb = sqrt( 4*log(1/delta2)/(n2*(e-2)) ) / Kmu
    gam_ub = 1/Kmu
    
    if gammastar > gam_ub :
        gammastar = gam_ub
    elif gammastar < gam_lb:
        gammastar = gam_lb
    else:
        gammastar = gammastar
    
    bound = mutandem_risk + gammastar * a + bprime / gammastar
    return min(bound , 1.0), gammastar


#Corollary 17 : \label{cor:bound_variance_grid}
def _varMUBernstein(vartandem_risk, n2, KL, mu=0.0, c1=1.0, delta1=0.05, unionBound = False):

    if unionBound:
        nu1  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
        nu1 = ceil(log(nu1)/log(c1))
    else:
        nu1 = 1

    # From the proof of Collorary 17.
    a = vartandem_risk
    bprime = c1*(2*KL + log(nu1) - log(delta1)) / (2*(n2-1))
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    # From the proof of Collorary 17.
    tstar2 = 1./ (sqrt(a/(Kmu**2 * bprime)+1)+1 )
    lambdastar = 2*(n2-1)*tstar2/n2
    
    # From the proof of Collorary 17. Equation (10)
    varMuBound = a / (1 - tstar2) + Kmu**2 * bprime / (tstar2 * (1 - tstar2))

    return varMuBound, lambdastar


# Optimize MU
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeMUBernstein(MVBounds, data, incl_oob, c1=1.0, c2=1.0, delta=0.05, options=None):
    options = dict() if options is None else options
    if 'mu_grid' not in options:
        """ problem to be fixed : options["mu"] """
        mutandemrisks, musquaretandem_risks, n2 = MVBounds.mutandem_risks(options["mu"], data, incl_oob)
        vartandemrisks = (n2/(n2-1))*(musquaretandem_risks / n2 - np.square(mutandemrisks / n2))
        return _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2,  c1=c1, c2=c2, mu=options["mu"], delta=delta, options=None)
    else:
        mu_grid = options['mu_grid']
        delta /= len(mu_grid)
        best_bound = (2,)
        for mu in mu_grid:
            mutandemrisks, musquaretandem_risks, n2 = MVBounds.mutandem_risks(mu, data, incl_oob)
            vartandemrisks = (n2 / (n2 - 1)) * (musquaretandem_risks / n2 - np.square(mutandemrisks / n2))
            b = _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2, mu=mu, c1=c1, c2=c2, delta=delta, options=None)
            if b[0] <= best_bound[0]:
                best_bound = b
            elif b[0] > best_bound[0]:
                # if stop improving, break
                break
        return best_bound


# Same as above, but mu is now a single value. If None, mu will be optimized
def _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2s, mu=None, c1=1.0, c2=1.0, delta=0.05, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    mu_input = mu

    if optimizer not in ['GD', 'RProp', 'iRProp']:
        warn('optimizeMU: unknown optimizer: \'' + optimizer + '\', using iRProp')
        optimizer = 'iRProp'

    m = mutandemrisks.shape[0]
    pi = uniform_distribution(m)
    rho = uniform_distribution(m)

    def _bound(rho, mu):  # Compute bound
        rho = softmax(rho)
        KL = kl(rho, pi)

        mutandemrisk = np.average(np.average(mutandemrisks / n2s, weights=rho, axis=1), weights=rho)
        vartandemrisk = np.average(np.average(vartandemrisks, weights=rho, axis=1), weights=rho)

        # Compute the bound for the true variance by Corollary 17
        varMuBound, lam = _varMUBernstein(vartandemrisk, np.min(n2s), KL, mu, c1=c1, delta1=delta / 2.)

        # Compute the bound of the muTandem loss by Corollary 20.
        muTandemBound, gam = _muBernstein(mutandemrisk, varMuBound, np.min(n2s), KL, mu, c2=c2, delta2=delta / 2.)

        bound =  muTandemBound / ((0.5 - mu) ** 2)

        return (bound, mu, lam, gam)

    def _gradient(rho, mu, lam, gam):
        n2 = np.min(n2s)
        # range factor
        Kmu = max(1 - mu, 1 - 2 * mu)

        a = (e-2)*gam / (1 - n2*lam/(2*(n2-1)))
        b = c2/(gam*n2) + (e-2)*gam*c1*Kmu**2 / (n2*lam*(1-n2*lam/(2*(n2-1))))

        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho, 1.0 - Srho))

        return 2 * np.dot(np.dot(mutandemrisks, Srho) + a * np.dot(vartandemrisks, Srho) + b * (1 + np.log(Srho / pi)), Smat)

    max_iterations = options.get('max_iterations', None)
    eps = options.get('eps', 10 ** -9)

    if optimizer == 'GD':
        def _optRho(rho, mu, lam, gam):
            lr = options.get('learning_rate', 1)
            return GD(lambda x: _gradient(x, mu, lam, gam), lambda x: _bound(x, mu)[0], rho, \
                      max_iterations=max_iterations, lr=lr, eps=eps)
    elif optimizer == 'RProp':
        max_iterations = options.get('max_iterations', 1000)

        def _optRho(rho, mu, lam, gam):
            return RProp(lambda x: _gradient(x, mu, lam, gam), rho, eps=eps, max_iterations=max_iterations)
    elif optimizer == 'iRProp':
        max_iterations = options.get('max_iterations', 100)

        def _optRho(rho, mu, lam, gam):
            return iRProp(lambda x: _gradient(x, mu, lam, gam), lambda x: _bound(x, mu)[0], rho, \
                          eps=eps, max_iterations=max_iterations)

    rho = uniform_distribution(m)
    b, mu, lam, gam = _bound(rho, mu=mu_input)
    bp = b + 1
    while abs(b - bp) > eps:
        bp = b
        # Optimize rho
        nrho = _optRho(rho, mu, lam, gam)
        # Optimize lam + gam
        b, nmu, nlam, ngam = _bound(nrho, mu=mu_input)
        if b > bp:
            b = bp
            break
        rho, mu, lam, gam = nrho, nmu, nlam, ngam
    return (min(1.0, b), softmax(rho), mu, lam, gam)

