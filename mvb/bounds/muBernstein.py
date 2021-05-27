#
# Implements the MU Bernstein bound.
#
import numpy as np
from math import log, sqrt, exp, e, pi, ceil, nan
from .tools import solve_kl_sup, solve_kl_inf, Lambert
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp


### Find mu^* by binary search
def MUBernstein(MVBounds, data, incl_oob, KL, mu_range = (-0.5, 0.5), lam=None, gam=None, delta=0.05):
    """ If lam and gam are provided, find the closest points in the grid to compute the bound """
    
    # calculate the bound for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandem_risk, vartandem_risk, n2 = MVBounds.mutandem_risk(mu, data, incl_oob)

        # Compute the bound for the variance
        varUB, _ = _varMUBernstein(vartandem_risk, n2, KL, mu, lam, delta1= delta/2., unionbound=True)

        # Compute the bound for mu-tandem loss
        bernTandemUB, _ = _muBernstein(mutandem_risk, varUB, n2, KL, mu, gam, delta1= delta/2., delta2= delta/2., unionbound=True)
  
        # Compute the overall bound
        bnd = bernTandemUB / (0.5-mu)**2
        return (bnd, mu, mutandem_risk, vartandem_risk, varUB, bernTandemUB)

    """ We need grid in all cases to:
        1. Consider the union bound (delta /= number)
        2. Find the closest mu_i in the grid for mu_star to compute the bound (no need here since the bound is already optimized using the grid)
    """
    # define the grids
    number = 200
    delta /= number
    
    if len(mu_range)==1:
        # nothing to be optimized.
        opt_bnd, opt_mu, opt_mutandem_risk, opt_vartandem_risk, opt_varUB, opt_bernTandemUB = _bound(mu_range[0])
    else:
        mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
        opt_bnd, opt_mu, opt_mutandem_risk, opt_vartandem_risk, opt_varUB, opt_bernTandemUB = Binary_Search(lambda x: _bound(x), mu_grid, 'mu')
    
    #print('final bound', opt_bnd)
    return (min(1.0, opt_bnd), (opt_mu,) , min(1.0, opt_mutandem_risk), min(1.0, opt_vartandem_risk), min(1.0, opt_varUB), min(1.0, opt_bernTandemUB))

# PAC-Bayes Bennett
def _muBernstein(mutandem_risk, varMuBound, n2, KL, mu=0.0, gam=None, c1=1.05, c2=1.05, delta1=0.05, delta2=0.05, unionbound=True):
    # range factor
    Kmu = max(1-mu, 1-2*mu)
    
    # E[varMuBound]<=Kmu^2/4
    a = min(varMuBound, Kmu**2/4)
    
    """ Bernstein bound """
    """
    nu2 = sqrt((e-2)*n2 / (4*log(1/delta2)))
    nu2 = ceil(log(nu2)/log(c2))

    b = c2*(2*KL  + log(nu2/delta2))/n2
    
    gam_star = sqrt(c2*b / ((e-2)*a))
    bound = mutandem_risk + (e-2)*gam_star*a + b/gam_star
    """
    
    """ Bernnett bound """
    # coefficient of the variance term : phi(Kmu*gam)/(gam*Kmu**2)
    def _VarCoeff(gam):
        return (e**(Kmu*gam) - Kmu*gam - 1) / (gam * Kmu**2)
    """
    # Bennett bound for a given gamma
    def _bound(gam):
        # From the proof of Collorary 20.
        first = _VarCoeff(gam) * a
        c = 1/e * (4/n2*log(1/delta2) - 1)
        f_inv = log(Kmu*gam/(1+W0(c))) / log(c2) + 1    
        #print('gam', gam, 'f_inv', f_inv)
        second = c2/(n2*gam) * (2*KL + log(pi**2/(6*delta2)) + 2*log(f_inv))
        return (mutandem_risk + first + second, gam)
    
    # gradient for the Bennett bound
    def _gradient(gam):
        print('grad')
        gam = gam[0]
        first = _VarCoeff(gam) * a
        
        c = 1/e * (4/n2*log(1/delta2) - 1)
        f_inv = log(Kmu*gam/(1+W0(c))) / log(c2) + 1
        second = c2/(n2*gam**2) * (2/(f_inv*log(c2)) - 2*log(f_inv) - 2*KL - log(pi**2/(6*delta2)))
        print('f_inv', f_inv, 'gam', gam, 'first',first, 'second', second)
        return first + second
    """
    """ obtain gam_star by either of the following methods """
    """ ## method 1:
        ## It should return the tightest bound for the union-gamma bound.
        ## Find gam_star by gradient descent of the union-gamma bound.
    gam_init = np.array([1.0])
    gam_star = iRProp(lambda x: _gradient(x), lambda x: _bound(x), gam_init, \
                          eps=1e-9, max_iterations=100)

    bound gamma = _bound(gam_star)
    """
    
    """ ## method 2:
        ## It's the alternative method for method 1.
        ## Find gam_star by initializing gam_init to be the solution of the fixed-gamma bound.
        ## Then adjust a bit by applying binary search for the grids around gam_init.
    b = (2*KL  -log(delta2))/n2
    c = 1/e * (b* Kmu**2/a - 1) # exact solution
    gam_init = (1+W0(c))/Kmu # initial guess
    c0 = 1/e * (4/n2*log(1/delta2) - 1) # lower bound for c
    gam_min = (1+W0(c0))/Kmu

    #print('------ gam_min', gam_min, 'f_inv', log(Kmu*gam_min/(1+W0(c0))) / log(c2) + 1, 'bound', _bound(gam_min)[0])    
    
    #print('gam_init', gam_init, 'f_inv', log(Kmu*gam_init/(1+W0(c0))) / log(c2) + 1, 'bound', _bound(gam_init)[0])
    
    gam_grid = np.linspace((gam_min+gam_init)/2, gam_init+0.5, 2000)
    bound, gam_star = Binary_Search(lambda x: _bound(x), gam_grid, 'gam')
    #print('binary', 'gam_star', gam_star, 'f_inv', log(Kmu*gam_star/(1+W0(c0))) / log(c2) + 1, 'bound', bound)
    
    if _bound(gam_min)[0]< bound:
        bound, gam_star = _bound(gam_min)
        #print('gam_min_has_min', end=' ')
    #print('gam_final', gam_star, 'f_inv', log(Kmu*gam_star/(1+W0(c0))) / log(c2) + 1)

    """
    
    """ ## method 3:
        ## Compute gam_star directly using the fixed-gamma bound.
        ## Then compute the bound by plugging in gam_star into the union-gamma bound.
    b = (2*KL  -log(delta2))/n2
    c = 1/e * (b* Kmu**2/a - 1)

    x_star = W0(c)
    gam_star = (1+x_star)/Kmu
    
    ## compute the bound by the either of the following two methods
    # method 1: computer the bound by plugging in the obtained gamma into the fixed-gamma bound, (ignore the effect of union bound)
    #bound = mutandem_risk + _VarCoeff(gam_star) * a + b / gam_star
    #pseudo_second = c2 * (2*KL  -log(6 * delta2/(pi**2 * 10)))/n2 / gam_star # just for test
    #bound = mutandem_risk + _VarCoeff(gam_star) * a + pseudo_second # test the effect of c2 # just for test
    
    # method 2: computer the bound by plugging in the obtained gamma into the union-gamma bound
    bound, gam_star = _bound(gam_star)
    """
    
    ## method 4:
    ## Cconsider the lower bound for the empirical bound for the variance (non-zero)
    ## such that we have \gamma_max
    
    if unionbound == True:
        # compute gamma_min
        c_min = 1/e * (4/n2*log(1/delta2) - 1)     
        gam_min = (1. + Lambert(c_min, 'W0'))/Kmu
        #print('gam_min', gam_min)
        
        # compute gamma_max
        nu1  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
        nu1 = ceil(log(nu1)/log(c1))

        alpha = 1./ (1+(n2-1)/(2*Kmu*c1*log(nu1/delta1)))
        c_max = -alpha * e**(-alpha)
        gam_max = - (Lambert(c_max, 'W-1')+alpha)/Kmu
        #print('gam_max', gam_max)
        
        # computer nu2
        nu2 = ceil(log(gam_max/gam_min)/log(c2))
        
        # construct the grid
        base = gam_min
        gam_grid = np.array([c2**i * base for i in range(nu2)])
    else:
        nu2 = sqrt(n2)
        
    # E[varMuBound]<=Kmu^2/4
    a = min(varMuBound, Kmu**2/4)
    b = 2*KL  + log(nu2/delta2)/n2
    
    if gam is None:
        # compute gam_star
        c = 1/e * (b* Kmu**2/a - 1)
        gam_star = (1+Lambert(c, 'W0'))/Kmu
    else:
        # gam_star is already provided by optimization
        gam_star = gam
        
    if unionbound == True:
        # find the closest lam_star in the grid to calculate the bound
        gam_star = gam_grid[np.argmin(abs(gam_grid-gam_star))]
        
    bound = mutandem_risk +  _VarCoeff(gam_star) * a + b / gam_star  
    
    return bound, gam_star


# Compute the bound for the variance
def _varMUBernstein(vartandem_risk, n2, KL, mu=0.0, lam=None, c1=1.05, delta1=0.05, unionbound=True):

    if unionbound == True:
        nu1  = 0.5 * sqrt( (n2-1)/log(1/delta1)+1 ) + 0.5
        nu1 = ceil(log(nu1)/log(c1))

        # construct the grid
        base = 2*(n2-1)/n2 * 1./(sqrt( (n2-1)/log(1/delta1)+1 ) + 1)
        lam_grid = np.array([c1**i * base for i in range(nu1)])
    else:
        nu1 = sqrt(n2)
    
    
    # From the proof of Collorary 17.
    a = vartandem_risk
    bprime = 2*KL + log(nu1) - log(delta1) / (2*(n2-1))
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)
    
    if lam is None:
        # compute lam_star
        t_star = 1./ (sqrt(a/(Kmu**2 * bprime)+1)+1 )
        lam_star = 2*(n2-1)*t_star/n2
    else:
        # lam_star is already provided by optimization
        lam_star = lam
        
    if unionbound == True:
        # find the closest lam_star in the grid to calculate the bound
        lam_star = lam_grid[np.argmin(abs(lam_grid-lam_star))]
        t_star = lam_star/2. * n2/(n2-1)
    
    # From the proof of Collorary 17. Equation (10)
    varMuBound = a / (1 - t_star) + Kmu**2 * bprime / (t_star * (1 - t_star))

    return varMuBound, lam_star


# Optimize MUBennett
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeMUBernstein(MVBounds, data, incl_oob, c1=1.05, c2=1.05, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    mu_range = options.get('mu_bern', (-0.5, 0.5))
    
    # calculate the optimized bound (over rho) for a given mu
    def _bound(mu):
        # Compute the quantities depend on mu
        mutandemrisks, musquaretandem_risks, n2 = MVBounds.mutandem_risks(mu, data, incl_oob)
        vartandemrisks = (n2 / (n2 - 1)) * (musquaretandem_risks / n2 - np.square(mutandemrisks / n2))
        
        # Return the optimized (over rho) bound for a given mu
        return _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2, mu=mu, c1=c1, c2=c2, delta=delta, abc_pi=abc_pi, options=options)
    
    # define the number of grids
    number = 200
    mu_grid = np.array([(mu_range[0]+(mu_range[1]-mu_range[0])/number * i) for i in range(number)])
    """ # Forget about the union bound during optimization. Turn on if needed. """
    delta /= number
    
    opt_bnd, opt_rho, opt_mu, opt_lam, opt_gam = Binary_Search(lambda x: _bound(x), mu_grid, 'mu')

    return (min(opt_bnd, 1.0), opt_rho, opt_mu, opt_lam, opt_gam)

# optimize over rho for a fixed mu
def _optimizeMUBernstein(mutandemrisks, vartandemrisks, n2s, mu=None, c1=1.05, c2=1.05, delta=0.05, abc_pi=None, options=None):
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

        # Compute the bound for the true variance by Corollary 17
        varMuBound, lam = _varMUBernstein(vartandemrisk, np.min(n2s), KL, mu, c1=c1, delta1=delta / 2., unionbound=False)

        # Compute the bound of the muTandem loss by Corollary 20.
        muTandemBound, gam = _muBernstein(mutandemrisk, varMuBound, np.min(n2s), KL, mu, c1=c1, c2=c2, delta1=delta / 2., delta2=delta / 2., unionbound=False)

        bound =  muTandemBound / ((0.5 - mu) ** 2)

        return (bound, mu, lam, gam)
    
    # gradient over rho
    def _gradient(rho, mu, lam, gam):
        n2 = np.min(n2s)
        # range factor
        Kmu = max(1 - mu, 1 - 2 * mu)
        phi = e**(Kmu*gam)-Kmu*gam-1
        
        """ # Forget about the union bound during optimization. """
        c1, c2 = 1., 1.
        a = phi/(gam*Kmu**2) / (1 - n2*lam/(2*(n2-1)))
        b = c2/(gam*n2) + phi/gam * c1 / (n2*lam*(1-n2*lam/(2*(n2-1))))

        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho, 1.0 - Srho))
        # avoid log(0)
        #Srho_p = np.where(Srho > 10 ** -10, Srho, -10)
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
        print("Use _MUBernsteinGrid")
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
                warn('The ' + obj + ' function might be non-convex! else')
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