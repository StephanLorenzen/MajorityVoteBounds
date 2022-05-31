#
# Implements the CCPBB bound.
#
import numpy as np
from scipy.special import lambertw
from math import log, sqrt, exp, e, pi, ceil, nan
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp


""" Calculate the CCPBB bound """
def CCPBB(MVBounds, data, incl_oob, KL, mu_opt = 0., lam=None, gam=None, delta=0.05):   
    # calculate the bound for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandem_risk, _, _, vartandem_risk, _, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)

        # Compute the empirical bound of the variance
        ub_var, _ = _VAR_bound(vartandem_risk, n2, KL, mu, lam, delta1= delta/2.)

        # Compute the empirical bound of the mu-tandem loss
        ub_mutandem, _ = _PBB_bound(mutandem_risk, ub_var, n2, KL, mu, gam, delta1= delta/2., delta2= delta/2.)
  
        # Compute the overall bound
        bnd = ub_mutandem / (0.5-mu)**2
        return (bnd, mutandem_risk, vartandem_risk, ub_var, ub_mutandem)
    
    opt_bnd, opt_mutandem_risk, opt_vartandem_risk, opt_ub_var, opt_ub_mutandem = _bound(mu_opt)

    return (min(1.0, opt_bnd) , min(1.0, opt_mutandem_risk), min(1.0, opt_vartandem_risk), min(1.0, opt_ub_var), min(1.0, opt_ub_mutandem))

""" Implement of the PAC-Bayes-Bennett bound """
def _PBB_bound(mutandem_risk, varMuBound, n2, KL, mu=0.0, gam=None, c1=1.05, c2=1.05, delta1=0.05, delta2=0.05):
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    def _var_coeff(gam):
        return (e**(Kmu*gam)-Kmu*gam-1) / (gam*Kmu**2)

    # compute gamma_min
    c_min = 1/e * (4*Kmu**2/(n2*Kmu**2) * log(1/delta2) - 1)     
    gam_min = (1. + lambertw(c_min, k=0).real)/Kmu
    
    # compute gamma_max
    Vmin = 2*Kmu**2*log(1/delta1)/(n2-1)
    alpha = 1./ (1 + Kmu**2/Vmin)
    gam_max = - (lambertw(-alpha * e**(-alpha), k=-1).real +alpha)/Kmu
    
    # compute k_gamma
    k_gamma = ceil(log(gam_max/gam_min)/log(c2))
    
    # compute k_lambda
    k_lambda  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
    k_lambda = ceil(log(k_lambda)/log(c1))

    # E[varMuBound]<=Kmu^2/4
    var = min(varMuBound, Kmu**2/4)
    comp = (2*KL  + log(k_gamma*k_lambda/delta2))/n2
    
    if gam is not None:
        # when the optimal gam is provided
        gam_star = gam
    else:
        # when the optimal gam is not known
        # find the optimal gam in the grid constructed by the original comp
        comp_origin = (2*KL  + log(1/delta2))/n2
    
        # construct the grid
        base = gam_min
        gam_grid = np.array([c2**i * base for i in range(k_gamma)])
        
        # compute gam_star
        gam_star = 1/e * (Kmu**2*comp_origin/var - 1)
        gam_star = (1. + lambertw(gam_star, k=0).real)/Kmu
        
        # find the closest gam_star in the grid to calculate the bound
        gam_star = gam_grid[np.argmin(abs(gam_grid-gam_star))]
        
    bound = mutandem_risk +  _var_coeff(gam_star) * var + comp / gam_star
    
    return bound, gam_star

""" Implementation of Theorem 13 in NeurIPS 2021 """
def _VAR_bound(vartandem_risk, n2, KL, mu=0.0, lam=None, c1=1.05, delta1=0.05):
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    k_lambda  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
    k_lambda = ceil(log(k_lambda)/log(c1))

    var = vartandem_risk
    # consider union bound over \lambda when calculating the bound
    comp = Kmu**2*(2*KL + log(k_lambda/delta1)) / (2*(n2-1))
    
    if lam is not None:
        # when the optimal lam is provided
        t_star = lam*n2/(2*(n2-1))
    else:
        # when the optimal lam is not known
        # find the optimal t_star in the grid constructed by the original comp
        comp_origin = Kmu**2*(2*KL + log(1/delta1)) / (2*(n2-1))

        # construct the grid for t = lam*n2/(2*(n2-1))
        base = 1./(sqrt( (n2-1)/log(1/delta1)+1 ) + 1)
        t_grid = np.array([c1**i * base for i in range(k_lambda)])
        
        # compute t_star
        t_star = 1./ (sqrt(var/comp_origin+1)+1 )
        
        # find the closest t_star in the grid to calculate the bound
        t_star = t_grid[np.argmin(abs(t_grid-t_star))]

    varMuBound = var / (1 - t_star) +  comp / (t_star * (1 - t_star))
    lam_star = 2*(n2-1)*t_star/n2

    return varMuBound, lam_star

""" Optimization of PAC-Bayes-Bennett """
def _PBB_opt(mutandem_risk, varMuBound, n2, KL, mu=0.0, c1=1.05, c2=1.05, delta1=0.05, delta2=0.05):
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    def _var_coeff(gam):
        return (e**(Kmu*gam)-Kmu*gam-1) / (gam*Kmu**2)

    # compute gamma_min
    c_min = 1/e * (4*Kmu**2/(n2*Kmu**2) * log(1/delta2) - 1)     
    gam_min = (1. + lambertw(c_min, k=0).real)/Kmu
    
    # compute gamma_max
    Vmin = 2*Kmu**2*log(1/delta1)/(n2-1)
    alpha = 1./ (1 + Kmu**2/Vmin)
    gam_max = - (lambertw(-alpha * e**(-alpha), k=-1).real +alpha)/Kmu
    
    # computer k_gamma
    k_gamma = ceil(log(gam_max/gam_min)/log(c2))
    
    # construct the grid
    base = gam_min
    gam_grid = np.array([c2**i * base for i in range(k_gamma)])

    # E[varMuBound]<=Kmu^2/4
    var = min(varMuBound, Kmu**2/4)
    comp = (2*KL  + log(1/delta2))/n2
    
    # compute gam_star
    gam_star = 1/e * ((1-mu)**4*comp/var - 1)
    gam_star = (1. + lambertw(gam_star, k=0).real)/Kmu
    
    # find the closest gam_star in the grid to calculate the bound
    gam_star = gam_grid[np.argmin(abs(gam_grid-gam_star))]
    bound = mutandem_risk +  _var_coeff(gam_star) * var + comp / gam_star
    
    return bound, gam_star

""" Optimization of Theorem 13 in NeurIPS 2021 """
def _VAR_opt(vartandem_risk, n2, KL, mu=0.0, c1=1.05, delta1=0.05):
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    k_lambda  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
    k_lambda = ceil(log(k_lambda)/log(c1))

    # construct the grid for t = lam*n2/(2*(n2-1))
    base = 1./(sqrt( (n2-1)/log(1/delta1)+1 ) + 1)
    t_grid = np.array([c1**i * base for i in range(k_lambda)])
    
    a = vartandem_risk
    b = Kmu**2*(2*KL + log(1/delta1)) / (2*(n2-1))
    
    # compute t_star
    t_star = 1./ (sqrt(a/b+1)+1 )
    
    # find the closest t_star in the grid to calculate the bound
    t_star = t_grid[np.argmin(abs(t_grid-t_star))]

    varMuBound = a / (1 - t_star) +  b / (t_star * (1 - t_star))
    lam_star = 2*(n2-1)*t_star/n2

    return varMuBound, lam_star

"""
Optimize CCPBB
options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
Default for opt is iRProp
"""
def optimizeCCPBB(MVBounds, data, incl_oob, c1=1.05, c2=1.05, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    
    # calculate the optimized bound (over rho) for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandemrisks, _, _, musquaretandem_risks, n2 = MVBounds.mutandem_risks(mu, data, incl_oob)
        vartandemrisks = (n2 / (n2 - 1)) * (musquaretandem_risks / n2 - np.square(mutandemrisks / n2))
        
        # Return the optimized (over rho) bound for a given mu
        return _optimizeCCPBB(mutandemrisks, vartandemrisks, n2, mu=mu, c1=c1, c2=c2, delta=delta, abc_pi=abc_pi, options=options)
    
    # define the number of grids
    mu_range = options.get('mu_CCPBB', (-0.5, 0.5))
    number = 400
    mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
    
    opt_bnd, opt_rho, opt_mu, opt_lam, opt_gam = Binary_Search(lambda x: _bound(x), mu_grid, 'mu')

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_lam, opt_gam)

"""
Optimize CCPBB for a fixed mu
"""
def _optimizeCCPBB(mutandemrisks, vartandemrisks, n2s, mu=None, c1=1.05, c2=1.05, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    mu_input = mu

    if optimizer not in ['GD', 'RProp', 'iRProp']:
        warn('optimizeMU: unknown optimizer: \'' + optimizer + '\', using iRProp')
        optimizer = 'iRProp'

    def _bound(rho, mu):  # Compute bound
        rho = softmax(rho)
        KL = kl(rho, pi)

        mutandemrisk = np.average(np.average(mutandemrisks / n2s, weights=rho, axis=1), weights=rho)
        vartandemrisk = np.average(np.average(vartandemrisks, weights=rho, axis=1), weights=rho)

        # Compute the empirical bound of the variance
        ub_var, lam = _VAR_opt(vartandemrisk, np.min(n2s), KL, mu, c1=c1, delta1=delta / 2.)

        # Compute the empirical bound of the mu-tandem loss
        ub_mutandem, gam = _PBB_opt(mutandemrisk, ub_var, np.min(n2s), KL, mu, c1=c1, c2=c2, delta1=delta / 2., delta2=delta / 2.)

        bound =  ub_mutandem / ((0.5 - mu) ** 2)

        return (bound, mu, lam, gam)
    
    # gradient over rho
    def _gradient(rho, mu, lam, gam):
        n2 = np.min(n2s)
        # range factor
        Kmu = max(1 - mu, 1 - 2 * mu)
        phi = e**(Kmu*gam)-Kmu*gam-1
        
        a = phi/(Kmu**2*gam) / (1 - n2*lam/(2*(n2-1)))
        b = 1/(gam*n2) + phi/(Kmu**2*gam) * Kmu**2 / (n2*lam*(1-n2*lam/(2*(n2-1))))

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


    m = mutandemrisks.shape[0]
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


# Implement the Binary Search Algorithm to find the mu^* and the corresponding bound
# func_bnd : the target function
# mu_grid : the grid of mu to search
# obj: Binary search for either mu or gam
def Binary_Search(func_bnd, mu_grid, obj):
    def _mean(a,b):
        return (a+b)/2
    
    # in case the binary search fails, use grid search instead
    def _Grid(grid):
        print("Use _CCPBBGrid")
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