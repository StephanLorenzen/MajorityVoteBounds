#-*- coding:utf-8 -*-
""" Computes the CCPBSkl bound. """

import numpy as np
from .tools import validate_inputs, xi, solve_kl_sup, solve_kl_inf
from math import log, sqrt
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 


def CCPBSkl(MVBounds, data, incl_oob, KL, mu_opt = 0., delta=0.05):   
    # calculate the bound for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mu_prime = mu**2 if mu>=0 else -mu*(1-mu) # mu_prime for Split-kl
        b = (1-mu)**2 #for Split-kl
        a = -mu*(1-mu) if mu>=0 else mu**2 #for Split-kl
        mutrisk, mutriskP, mutriskM, _, _, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)
        
        # Split-kl for mutandem risk
        rhs = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
        ub_plus = solve_kl_sup(mutriskP/(b-mu_prime), rhs)
        lb_minus = 0. if mu_prime==a else solve_kl_inf(mutriskM/(mu_prime-a), rhs)
        ub_mutandem = mu_prime + (b-mu_prime)*ub_plus - (mu_prime-a)*lb_minus

        # Compute the overall bound
        bnd = ub_mutandem / (0.5-mu)**2
        return (bnd, mutriskP, mutriskM, ub_mutandem)
    
    opt_bnd, opt_mutriskP, opt_mutriskM, opt_ub_mutandem = _bound(mu_opt)
    b = (1-mu_opt)**2

    return (min(1.0, opt_bnd) , opt_mutriskP, opt_mutriskM, min(b, opt_ub_mutandem))


def optimizeCCPBSkl(MVBounds, data, incl_oob, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    
    def _bound(mu):
        # Compute the quantities depend on mu
        _, mutrisksP, mutrisksM, _, n2s = MVBounds.mutandem_risks(mu, data, incl_oob)
        
        # Return the optimized (over rho) bound for a given mu
        return _optimizeCCPBSkl(mutrisksP, mutrisksM, n2s, mu=mu, delta=delta, abc_pi=abc_pi, options=options)
    
    # define the number of grids
    mu_range = options.get('mu_CCPBSkl', (-0.5, 0.5))
    number = 400
    mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
    
    opt_bnd, opt_rho, opt_mu, opt_lam, opt_gam = Binary_Search(lambda x: _bound(x), mu_grid, 'mu')

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_lam, opt_gam)

"""
Optimize CCPBSkl for a fixed mu
"""
def _optimizeCCPBSkl(mutrisksP, mutrisksM, n2s, mu=None, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    mu_input = mu

    if optimizer not in ['GD', 'RProp', 'iRProp']:
        warn('optimizeMU: unknown optimizer: \'' + optimizer + '\', using iRProp')
        optimizer = 'iRProp'

    def _bound(rho, mu, lam=None, gam=None):  # Compute bound
        n2 = np.min(n2s)
        rho = softmax(rho)
        KL = kl(rho, pi)

        # Compute the quantities depend on mu
        mu_prime = mu**2 if mu>=0 else -mu*(1-mu) # mu_prime for Split-kl
        b_range = (1-mu)**2 #for Split-kl
        a_range = -mu*(1-mu) if mu>=0 else mu**2 #for Split-kl
        mutriskP = np.average(np.average(mutrisksP / n2s, weights=rho, axis=1), weights=rho)
        mutriskM = np.average(np.average(mutrisksM / n2s, weights=rho, axis=1), weights=rho)
        
        losstermP = mutriskP/(b_range-mu_prime)
        losstermM = 0 if mu_prime==a_range else mutriskM/(mu_prime-a_range)
        
        # upper bound of losstermP
        if lam is None:
            lam = 2.0 / (sqrt((2.0*n2*losstermP)/(2*KL+log(4.0*sqrt(n2)/delta)) + 1) + 1)
        ub_plus = losstermP/(1-lam/2)+(2*KL+log(4*sqrt(n2)/delta))/(lam*(1-lam/2)*n2)
        # upper bound of losstermM
        if gam is None:
            gam = 2.0 if losstermM==0 else min(2.0, sqrt( (4.0*KL+log(16.0*n2/delta**2)) / (n2*losstermM) ))
        lb_minus = max(0.0, (1-gam/2.0)*losstermM-(2.0*KL+log(4*sqrt(n2)/delta))/(gam*n2))
        
        ub_mutandem = mu_prime + (b_range-mu_prime)*ub_plus - (mu_prime-a_range)*lb_minus
        bound =  ub_mutandem / ((0.5 - mu) ** 2)

        return (bound, mu, lam, gam)
    
    # gradient over rho
    def _gradient(rho, mu, lam, gam):
        n2 = np.min(n2s)
        mu_prime = mu**2 if mu>=0 else -mu*(1-mu) # mu_prime for Split-kl
        b_range = (1-mu)**2 #for Split-kl
        a_range = -mu*(1-mu) if mu>=0 else mu**2 #for Split-kl
        
        a = 1.0/(1.0-lam/2.0)
        b = (1-gam/2.0)
        c = (b_range-mu_prime)/(lam*(1.0-lam/2.0)*n2) + (mu_prime-a_range)/(gam*n2)

        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho, 1.0 - Srho))

        return 2*np.dot(a*np.dot(mutrisksP, Srho)-b*np.dot(mutrisksM, Srho)+c*(1+np.log(Srho/pi)), Smat)

    max_iterations = options.get('max_iterations', None)
    eps = options.get('eps', 10 ** -9)

    if optimizer == 'GD':
        def _optRho(rho, mu, lam, gam):
            lr = options.get('learning_rate', 1)
            return GD(lambda x: _gradient(x, mu, lam, gam), lambda x: _bound(x,mu,lam,gam)[0], rho, \
                      max_iterations=max_iterations, lr=lr, eps=eps)
    elif optimizer == 'RProp':
        max_iterations = options.get('max_iterations', 1000)

        def _optRho(rho, mu, lam, gam):
            return RProp(lambda x: _gradient(x, mu, lam, gam), rho, eps=eps, max_iterations=max_iterations)
    elif optimizer == 'iRProp':
        max_iterations = options.get('max_iterations', 100)

        def _optRho(rho, mu, lam, gam):
            return iRProp(lambda x: _gradient(x, mu, lam, gam), lambda x: _bound(x,mu,lam,gam)[0], rho, \
                          eps=eps, max_iterations=max_iterations)


    m = mutrisksP.shape[0]
    pi = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    rho = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
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
    return (b, softmax(rho), mu, lam, gam)

def Grid(func_bnd, grid, obj):
    #print("Use _CCPBSklGrid")
    opt_bnd = (2000.0,)
    for mu in grid:
        grid_bound = func_bnd(mu)
        if grid_bound[0] <= opt_bnd[0]:
            opt_bnd = grid_bound      
    return opt_bnd

def Binary_Search(func_bnd, mu_grid, obj):
    def _mean(a,b):
        return (a+b)/2
    
    # in case the binary search fails, use grid search instead
    def _Grid(grid):
        print("Use _CCPBSklGrid")
        opt_bnd = (2000.0,)
        for mu in grid:
            grid_bound = func_bnd(mu)
            if grid_bound[0] <= opt_bnd[0]:
                opt_bnd = grid_bound      
        return opt_bnd
        
    # initialize 4 points
    number = len(mu_grid)
    left, midleft, midright, right = 0, int(number/4), int(number*(3/4)), number -1
    bnd_left, bnd_midleft, bnd_midright, bnd_right = func_bnd(mu_grid[left]), func_bnd(mu_grid[midleft]), func_bnd(mu_grid[midright]), func_bnd(mu_grid[right])
    bnd_star = None

    while (right - left > 3 and bnd_star is None):
        # if the midleft bound is larger than the midright bound
        if bnd_midleft[0] > bnd_midright[0]:
            # if the left bound is even larger
            # then the bound is decreasing for the left three points
            if bnd_left[0] >= bnd_midleft[0]:
                # if the right bound is larger than the midright bound
                # then the minimum is somewhere around midright
                # we can ignore the leftest interval
                if bnd_right[0] >= bnd_midright[0]:
                    left, bnd_left = midleft, bnd_midleft
                    midleft, bnd_midleft = midright, bnd_midright
                    midright = int(_mean(midright, right))
                    bnd_midright = func_bnd(mu_grid[midright])
                else:
                    # the bnd_right is the minimum
                    warn('The rightest ' + obj + ' has the minimum!')
                    bnd_star = bnd_right
            else:
                # the function might be non-convex
                warn('The ' + obj + ' function might be non-convex! bnd_midleft[0] > bnd_midright[0]')
                bnd_star = _Grid(mu_grid[left:right+1])
        # if the midright bound is larger than the midleft bound
        elif bnd_midleft[0] < bnd_midright[0]:
            if bnd_right[0] >= bnd_midright[0]:
                if bnd_left[0] >= bnd_midleft[0]:
                    right, bnd_right = midright, bnd_midright
                    midright, bnd_midright = midleft, bnd_midleft
                    midleft = int(_mean(left, midleft))
                    bnd_midleft = func_bnd(mu_grid[midleft])
                else:
                    warn('The leftest ' + obj + ' has the minimum!')
                    bnd_star = bnd_left
            else:
                warn('The ' + obj + ' function might be non-convex! bnd_midleft[0] < bnd_midright[0]')
                bnd_star = _Grid(mu_grid[left:right+1])
        # if the middle two points are equal
        else:
            #print('in else')
            # if the left and the right bounds are larger
            if (bnd_left[0] > bnd_midleft[0] and bnd_right[0] > bnd_midright[0]):
                # if there is no other points in between midleft and midright
                # assign either bnd_midleft or bnd_midright to be bnd_star
                if midleft == midright:
                    bnd_star = bnd_left
                else:
                    # shift one of the middle points to keep searching
                    midleft = int(_mean(midleft, midright))
                    bnd_midleft = func_bnd(mu_grid[midleft])
            # the function is flat
            elif (bnd_left[0] == bnd_midleft[0] and bnd_right[0] == bnd_midright[0]):
                bnd_star = bnd_left
            # the function is not convex
            else:
                bnd_star = _Grid(mu_grid[left:right+1])

    if bnd_star is not None:
        bnd_star = bnd_star
    elif right - left <= 3:
        if bnd_midleft[0] <= bnd_midright[0]:
            bnd_star = bnd_midleft
        else:
            bnd_star = bnd_midright
    else:
        warn('Optimization is not completed!')
    return bnd_star