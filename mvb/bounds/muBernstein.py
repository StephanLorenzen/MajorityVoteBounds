#
# Implements the MU Bernstein bound.
#
import numpy as np
from math import log, sqrt, exp, e, ceil, nan
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp


### Find mu^* by binary search
def MUBernstein(MVBounds, data, incl_oob, KL, mu_range = (-5, 0.5), delta=0.05):
    
    # calculate the bound for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandem_risk, vartandem_risk, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)

        # Compute a bound over the variance from Corollary 17
        varUB, _ = _varMUBernstein(vartandem_risk, n2, KL, mu, delta1= delta/2.)

        # Plug the bound over variance to compute a bound over the muTandem loss following Corollary 20.
        bernTandemUB, _ = _muBernstein(mutandem_risk, varUB, n2, KL, mu, delta2= delta/2.)
  
        # Compute the overall bound
        bnd = bernTandemUB / (0.5-mu)**2
        return (bnd, mu, mutandem_risk, vartandem_risk, varUB, bernTandemUB)
    
    # optimized the bound over mu_grid by exhausting search
    def _MUBernsteinGrid(mu_grid):
        print("Use _MUBernsteinGrid")
        opt_bnd = (2000.0,)
        for mu in mu_grid:
            grid_bound = _bound(mu)
            if grid_bound[0] <= opt_bnd[0]:
                opt_bnd = grid_bound      
        return opt_bnd
        
    if len(mu_range)==1:
        # nothing to be optimized.
        opt_bnd, opt_mu, opt_mutandem_risk, opt_vartandem_risk, opt_varUB, opt_bernTandemUB = _bound(mu_range[0])
    else:
        # define the number of grids
        number = 1000
        mu_grid = [(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)]
        delta /= number
        opt_bnd, opt_mu, opt_mutandem_risk, opt_vartandem_risk, opt_varUB, opt_bernTandemUB = Binary_Search(lambda x: _bound(x), lambda x: _MUBernsteinGrid(x), mu_grid)

    return (min(1.0, opt_bnd), (opt_mu,) , min(1.0, opt_mutandem_risk), min(1.0, opt_vartandem_risk), min(1.0, opt_varUB), min(1.0, opt_bernTandemUB))

#Corollary 20 : \label{cor:pac-bayes-bernstein_grid}
def _muBernstein(mutandem_risk, varMuBound, n2, KL, mu=0.0, c2=1.0, delta2=0.05, unionBound = False):

    if unionBound:
        nu2 = ceil( log( sqrt( (1/2)*n2 / (4*log(1/delta2)) ) ) / log(c2) )
        #nu2 = ceil( log( sqrt( (e-2)*n2 / (4*log(1/delta2)) ) ) / log(c2) )
    else:
        nu2 = 1
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)
    
    # E[varMuBound]<=Kmu^2/4
    varMuBound = min(varMuBound, Kmu**2/4)
    
    # From the proof of Collorary 20.
    bprime=c2*(2*KL + log(nu2) -log(delta2))/n2
    #a=(e-2)*varMuBound
    a=(1/2)*varMuBound
    gammastar = sqrt(bprime / a)
    
    # The range of Gamma^*
    gam_lb = sqrt( 4*log(1/delta2)/(n2*(1/2)) ) / Kmu
    #gam_lb = sqrt( 4*log(1/delta2)/(n2*(e-2)) ) / Kmu
    gam_ub = 1/Kmu
    
    if gammastar > gam_ub :
        gammastar = gam_ub
    elif gammastar < gam_lb:
        gammastar = gam_lb
    else:
        gammastar = gammastar
    
    bound = mutandem_risk + gammastar * a + bprime / gammastar
    return bound, gammastar


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
    mu_range = options.get('mu_bern', (-5, 0.5))
    
    # calculate the optimized bound (over rho) for a given mu
    def _fix_mu_bound(mu):
        # Compute the quantities depend on mu
        mutandemrisks, musquaretandem_risks, n2 = MVBounds.mutandem_risks(mu, data, incl_oob)
        vartandemrisks = (n2 / (n2 - 1)) * (musquaretandem_risks / n2 - np.square(mutandemrisks / n2))
        
        # Return the optimized (over rho) bound for a given mu
        return _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2, mu=mu, c1=c1, c2=c2, delta=delta, options=None)
    
    # optimized the bound over mu_grid by exhausting search
    def _MUBernsteinGrid(mu_grid):
        print("Use _MUBernsteinGrid")
        opt_bnd = (2000.0,)
        for mu in mu_grid:
            grid_bound = _fix_mu_bound(mu)
            if grid_bound[0] <= opt_bnd[0]:
                opt_bnd = grid_bound      
        return opt_bnd
    
    # define the number of grids
    number = 1000
    mu_grid = [(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)]
    delta /= number    
    opt_bnd, opt_rho, opt_mu, opt_lam, opt_gam = Binary_Search(lambda x: _fix_mu_bound(x), lambda x: _MUBernsteinGrid(x), mu_grid)

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_lam, opt_gam)

# optimize over rho for a fixed mu
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
    return (b, softmax(rho), mu, lam, gam)

# Implement the Binary Search Algorithm to find the mu^* and the corresponding bound
# func_bnd : the target function
# grid_func : in case the binary search fails, use grid search instead
# mu_grid : the grid of mu to search
def Binary_Search(func_bnd, grid_func, mu_grid):

    def _mean(a,b):
        return (a+b)/2
        
    # initialize 4 points
    number = len(mu_grid)
    left, midleft, midright, right = 0, int(number/4), int(number*(3/4)), number -1
    bnd_left, bnd_midleft, bnd_midright, bnd_right = func_bnd(mu_grid[left]), func_bnd(mu_grid[midleft]), func_bnd(mu_grid[midright]), func_bnd(mu_grid[right])
    bnd_star = None

    while (right - left > 3 and bnd_star is None):
        #print('right-left', right-left)
        #print('bnd_left', np.round(bnd_left[0], 4), 'bnd_midleft', np.round(bnd_midleft[0], 4), 'bnd_midright', np.round(bnd_midright[0], 4), 'bnd_right', np.round(bnd_right[0], 4))
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
                    warn('The rightest mu has the minimum!')
                    bnd_star = bnd_right
            else:
                # the function might be non-convex
                warn('The function might be non-convex! bnd_midleft[0] > bnd_midright[0]')
                bnd_star = grid_func(mu_grid[left:right+1])
        # if the midright bound is larger than the midleft bound
        elif bnd_midleft[0] < bnd_midright[0]:
            if bnd_right[0] >= bnd_midright[0]:
                if bnd_left[0] >= bnd_midleft[0]:
                    right, bnd_right = midright, bnd_midright
                    midright, bnd_midright = midleft, bnd_midleft
                    midleft = int(_mean(left, midleft))
                    bnd_midleft = func_bnd(mu_grid[midleft])
                else:
                    warn('The leftest mu has the minimum!')
                    bnd_star = bnd_left
            else:
                warn('The function might be non-convex! bnd_midleft[0] < bnd_midright[0]')
                bnd_star = grid_func(mu_grid[left:right+1])
        # if the middle two points are equal
        else:
            # if the left and the right bounds are larger
            if (bnd_left[0] > bnd_midleft[0] and bnd_right[0] > bnd_midright[0]):
                # if there is no other points in between midleft and midright
                # assign either bnd_left or bnd_right to be bnd_star
                if midleft +1 == midright:
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
                warn('The function might be non-convex! else')
                bnd_star = grid_func(mu_grid[left:right+1])

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
