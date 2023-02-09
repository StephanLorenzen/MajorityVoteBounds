#
# Implements the CCPBUB bound.
#
import numpy as np
from scipy.special import lambertw
from math import log, sqrt, exp, e, pi, ceil, nan
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp


""" Calculate the CCPBUB bound """
def CCPBUB(MVBounds, data, incl_oob, KL, mu_opt = 0., gam=None, delta=0.05):   
    # calculate the bound for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandem_risk, _, _, _, muSQtandem_risk, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)

        # Compute the empirical bound of the mu-tandem loss
        ub_mutandem, _ = _PBUB_bound(mutandem_risk, muSQtandem_risk, n2, KL, mu, gam, delta= delta)
  
        # Compute the overall bound
        bnd = ub_mutandem / (0.5-mu)**2
        return (bnd, mutandem_risk, muSQtandem_risk, ub_mutandem)
    
    opt_bnd, opt_mutandem_risk, opt_muSQtandem_risk, opt_ub_mutandem = _bound(mu_opt)

    return (min(1.0, opt_bnd) , opt_mutandem_risk, opt_muSQtandem_risk, opt_ub_mutandem)

""" Implement of the PAC-Bayes Unexpected Bernstein bound (Mhammedi et al., 2019)"""
def _PBUB_bound(mutandem_risk, muSQtandem_risk, n2, KL, mu=0.0, gam=None, delta=0.05):
    b_range = (1-mu)**2
    def _var_coeff(gam, b):
        return (-gam*b-log(1-gam*b))/(gam*b**2)
    
    # compute k_gamma
    k_gamma = ceil(log(sqrt(n2/log(1/delta))/2)/log(2))
    # complexity
    comp = (2*KL  + log(k_gamma/delta))/n2
    # optimal gamma, if gam is not given, then the bound makes little sense
    gam_star = gam if gam is not None else 0.5
        
    bound = mutandem_risk +  _var_coeff(gam_star, b_range) * muSQtandem_risk + comp / gam_star
    
    return bound, gam_star

""" Optimization of PAC-Bayes Unexpected Bernstein """
def _PBUB_opt(mutandem_risk, muSQtandem_risk, n2, KL, mu=0.0, delta=0.05):
    b_range = (1-mu)**2
    def _var_coeff(gam, b):
        return (-gam*b-log(1-gam*b))/(gam*b**2)

    # compute k_gamma
    k_gamma = ceil(log(sqrt(n2/log(1/delta))/2)/log(2))
    # construct the grid of gamma
    gam_grid = np.array([1/(2**(i+1)*b_range) for i in range(k_gamma)])
    # complexity
    comp = (2*KL  + log(k_gamma/delta))/n2
    
    # grid search
    opt_bnd = (2000.0, 0.5)
    for gam in gam_grid:
        (grid_bound, opt_gam) = (mutandem_risk + _var_coeff(gam, b_range) * muSQtandem_risk + comp / gam, gam)
        if grid_bound <= opt_bnd[0]:
            opt_bnd = (grid_bound, opt_gam)     

    return opt_bnd[0], opt_bnd[1]

"""
Optimize CCPBUB
options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
Default for opt is iRProp
"""
def optimizeCCPBUB(MVBounds, data, incl_oob, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    
    # calculate the optimized bound (over rho) for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandemrisks, _, _, musquaretandem_risks, n2s = MVBounds.mutandem_risks(mu, data, incl_oob)
        
        # Return the optimized (over rho) bound for a given mu
        return _optimizeCCPBUB(mutandemrisks, musquaretandem_risks, n2s, mu=mu, delta=delta, abc_pi=abc_pi, options=options)
    
    # define the number of grids
    mu_range = options.get('mu_CCPBUB', (-0.5, 0.5))
    number = 400
    mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
    
    opt_bnd, opt_rho, opt_mu, opt_gam = Binary_Search(lambda x: _bound(x), mu_grid, 'mu')

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_gam)

"""
Optimize CCPBUB for a fixed mu
"""
def _optimizeCCPBUB(mutandemrisks, musquaretandem_risks, n2s, mu=None, delta=0.05, abc_pi=None, options=None):
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
        muSQtandemrisk = np.average(np.average(musquaretandem_risks / n2s, weights=rho, axis=1), weights=rho)

        # Compute the empirical bound of the mu-tandem loss
        ub_mutandem, gam = _PBUB_opt(mutandemrisk, muSQtandemrisk, np.min(n2s), KL, mu, delta=delta)

        bound =  ub_mutandem / ((0.5 - mu) ** 2)

        return (bound, mu, gam)
    
    # gradient over rho
    def _gradient(rho, mu, gam):
        n2 = np.min(n2s)
        b_range = (1-mu)**2
        def _var_coeff(gam, b):
            return (-gam*b-log(1-gam*b))/(gam*b**2)
        
        a = _var_coeff(gam, b_range)
        b = 1/(gam*n2)

        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho, 1.0 - Srho))

        return 2 * np.dot(np.dot(mutandemrisks, Srho) + a * np.dot(musquaretandem_risks, Srho) + b * (1 + np.log(Srho / pi)), Smat)

    max_iterations = options.get('max_iterations', None)
    eps = options.get('eps', 10 ** -9)

    if optimizer == 'GD':
        def _optRho(rho, mu, gam):
            lr = options.get('learning_rate', 1)
            return GD(lambda x: _gradient(x, mu, gam), lambda x: _bound(x, mu)[0], rho, \
                      max_iterations=max_iterations, lr=lr, eps=eps)
    elif optimizer == 'RProp':
        max_iterations = options.get('max_iterations', 1000)

        def _optRho(rho, mu, gam):
            return RProp(lambda x: _gradient(x, mu, gam), rho, eps=eps, max_iterations=max_iterations)
    elif optimizer == 'iRProp':
        max_iterations = options.get('max_iterations', 100)

        def _optRho(rho, mu, gam):
            return iRProp(lambda x: _gradient(x, mu, gam), lambda x: _bound(x, mu)[0], rho, \
                          eps=eps, max_iterations=max_iterations)


    m = mutandemrisks.shape[0]
    pi = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    rho = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    b, mu, gam = _bound(rho, mu=mu_input)
    bp = b + 1
    while abs(b - bp) > eps:
        bp = b
        # Optimize rho
        nrho = _optRho(rho, mu, gam)
        # Optimize lam + gam
        b, nmu, ngam = _bound(nrho, mu=mu_input)
        if b > bp:
            b = bp
            break
        rho, mu, gam = nrho, nmu, ngam
    return (b, softmax(rho), mu, gam)


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