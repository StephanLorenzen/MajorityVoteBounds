#
# Implements the MU bound.
#
import numpy as np
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 

# Implementation of MU
def MU(tandem_risk, gibbs_risk, n, n2, KL, mu_range = (-0.5, 0.5), delta=0.05):
    #calculate the bound for a given mu
    def _bound(mu):
        # UpperBound_TandemRisk by inverse kl
        rhs_tr = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
        ub_tr  = solve_kl_sup(tandem_risk, rhs_tr)
        
        # LowerBound_GibbsRisk by inverse kl
        rhs_gr = ( KL + log(4.0*sqrt(n)/delta) ) / n
        lb_gr  = solve_kl_inf(gibbs_risk, rhs_gr)
        
        # if mu == None, calculate mu by the close-form formula
        if mu is None:
            mu = (0.5*lb_gr - ub_tr)/(0.5-lb_gr)

        # bound
        muTandemUB = ub_tr - 2*mu*lb_gr + mu**2
        bnd = muTandemUB / (0.5-mu)**2
        return (bnd, mu, muTandemUB)
    
    """ We need grid in all cases to:
        1. Consider the union bound (delta /= number)
        2. Find the closest mu_i in the grid for mu_star to compute the bound
    """
    # define the grids
    number = 200
    delta /= number
    
    if len(mu_range)==1:
        # nothing to be optimized.
        mu_star = mu_range[0]
        # find the closest mu_i in the grid
        mu_grid = np.array([(-0.5+(0.5 - (-0.5))/number * i) for i in range(number)])
        mu_i = mu_grid[np.argmin(abs(mu_grid-mu_star))]
        opt_bnd, opt_mu, opt_muTandemUB = _bound(mu_i)
    else:
        
        """ # Implemented by the closed-form solution """
        _, mu_star, _ = _bound(mu=None)
        # find the closest mu_i in the grid
        mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
        mu_i = mu_grid[np.argmin(abs(mu_grid-mu_star))]
        opt_bnd, opt_mu, opt_muTandemUB = _bound(mu_i)
        
    return (min(1.0, opt_bnd), (opt_mu,) , min(1.0, opt_muTandemUB))

# Optimize MU
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeMU(tandem_risks, gibbs_risks, n, n2, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    mu_range = options.get('mu_kl', (-0.5, 0.5))
    
    # calculate the optimized bound (over rho) for a given mu
    def _bound(mu): 
        return _optimizeMU(tandem_risks, gibbs_risks, n, n2, mu=mu, delta=delta, abc_pi=abc_pi, options=options)

    number = 200
    mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
    """ # Forget about the union bound during optimization. Turn on if needed. """
    delta /= number
       
    """ # Implemented by the closed-form solution """
    opt_bnd, opt_rho, opt_mu, opt_lam, opt_gam = _bound(mu=None)
    

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_lam, opt_gam)

# Same as above, but mu is now a single value. If None, mu will be optimized
def _optimizeMU(tandem_risks, gibbs_risks, n, n2, mu=None, abc_pi=None, delta=0.05, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    mu_input = mu

    if optimizer not in ['GD', 'RProp', 'iRProp']:
        warn('optimizeMU: unknown optimizer: \''+optimizer+'\', using iRProp')
        optimizer = 'iRProp'

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

        # compute mu_star by ub_tnd and lb_gr if mu is not given
        if mu is None:
            mu = (0.5*lb_gr - ub_tnd)/(0.5-lb_gr)
         
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

    m = gibbs_risks.shape[0]
    pi  = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    rho = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    b, mu, lam, gam  = _bound(rho, mu=mu_input)
    bp = b+1
    while abs(b-bp) > eps:
        bp = b
        # Optimize rho
        nrho = _optRho(rho,mu,lam,gam)
        """ Optimize lam + gam ( also mu if mu_input is None; otherwise, mu is fixed to be mu_input) """
        b, nmu, nlam, ngam = _bound(nrho, mu=mu_input)
        if b > bp:
            b = bp
            break
        rho, mu, lam, gam = nrho, nmu, nlam, ngam
    return (b, softmax(rho), mu, lam, gam)

# Implement the Binary Search Algorithm to find the mu^* and the corresponding bound
# func_bnd : the target function
# mu_grid : the grid of mu to search
def Binary_Search(func_bnd, mu_grid):
    def _mean(a,b):
        return (a+b)/2
    
    # in case the binary search fails, use grid search instead
    def _Grid(grid):
        print("Use _MUklGrid")
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
        #print('right-left', right-left)
        #print('left', left, 'midleft', midleft, 'midright', midright, 'right', right)
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
                warn('The mu function might be non-convex! bnd_midleft[0] > bnd_midright[0]')
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
                    warn('The leftest mu has the minimum!')
                    bnd_star = bnd_left
            else:
                warn('The mu function might be non-convex! bnd_midleft[0] < bnd_midright[0]')
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
                warn('The mu function might be non-convex! else')
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