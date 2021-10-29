#
# Implements the TND and DIS bounds, as well as the CTD bound (a new version of the C-bound).
#

import numpy as np
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 

# Implementation of TND
def TND(tandem_risk, n, KL, delta=0.05):
    rhs   = ( 2.0*KL + log(2.0*sqrt(n)/delta) ) / n
    ub_tr = min(0.25, solve_kl_sup(tandem_risk, rhs))
    return 4*ub_tr

# Implementation of DIS
def DIS(gibbs_risk, disagreement, n, n2, KL, delta=0.05):
    g_rhs = ( KL + log(4.0*sqrt(n)/delta) ) / n
    g_ub  = min(1.0, solve_kl_sup(gibbs_risk, g_rhs))
    
    d_rhs = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
    d_lb  = solve_kl_inf(disagreement, d_rhs)
    return min(1.0, 4*g_ub - 2*d_lb)

# Implementation of CTD
def CTD(gibbs_risk, tandem_risk, n, n2, KL, delta=0.05):
    """ CTD bound (tandem risk version of the C bound)
    """
    if gibbs_risk > 0.5:
        return 1.0

    rhs_tr = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
    ub_tr  = solve_kl_sup(tandem_risk, rhs_tr)
    lb_tr  = solve_kl_inf(tandem_risk, rhs_tr)
    
    rhs_g = ( KL + log(4.0*sqrt(n)/delta) ) / n
    ub_g  = solve_kl_sup(gibbs_risk, rhs_g)
    lb_g  = solve_kl_inf(gibbs_risk, rhs_g)
   
    if lb_tr-ub_g+0.25 <= 0:
        return 1.0

    return min(1.0, (ub_tr-lb_g**2)/(lb_tr-ub_g+0.25))

# Optimize TND
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeTND(tandem_risks, n2, delta=0.05, abc_pi=None, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    
    if optimizer not in ['CMA', 'GD', 'RProp', 'iRProp']: # 'Alternate' remove for now
        warn('optimizeMV: unknown optimizer: \''+optimizer+'\', using GD')
        optimizer = 'GD'
    
    m   = tandem_risks.shape[0]
    rho = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    pi  = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)

    # Some helper functions
    def _tndr(rho): # Compute tandem risk from tandem risk matrix and rho
        return np.average(np.average(tandem_risks, weights=rho, axis=0), weights=rho)
    def _bound(rho, lam=None): # Compute value of bound (also optimize lambda if None)
        rho  = softmax(rho)
        tndr = _tndr(rho)
        KL   = kl(rho,pi)
        if lam is None:
            lam = 2.0 / (sqrt((2.0*n2*tndr)/(2.0*KL+log(2.0*sqrt(n2)/delta)) + 1) + 1)
        bound = tndr / (1.0 - lam/2.0) + (2.0*KL+log(2.0*sqrt(n2)/delta))/(lam*(1.0-lam/2.0)*n2)
        return (bound, lam) 

    if optimizer=='CMA':
        # Optimize with CMA
        import cma
        def _func(r):
            return _bound(r)[0]
        sol = cma.fmin(_func, rho, options.get('sigma', 0.1), {'verbose': -9})
        b, lam = _bound(sol[0])
        return (min(1.0,4*b), softmax(sol[0]), lam)
    else:
        # 1st order methods
        def _gradient(rho, lam): # Gradient (using rho = softmax(rho'))
            Srho = softmax(rho)
            # D_jS_i = S_i(1[i==j]-S_j)
            Smat = -np.outer(Srho, Srho)
            np.fill_diagonal(Smat, np.multiply(Srho,1.0-Srho))
            return np.dot(2*(np.dot(tandem_risks,Srho)+1.0/(lam*n2)*(1+np.log(Srho/pi))),Smat)

        max_iterations = options.get('max_iterations', None)
        eps = options.get('eps', 10**-9)
        
        if optimizer=='GD': # Gradient descent with adaptive lr
            def _optRho(rho, lam):
                lr = options.get('learning_rate', 1)
                return GD(lambda x: _gradient(x,lam), lambda x: _bound(x,lam)[0], rho,\
                        max_iterations=max_iterations, lr=lr, eps=eps)
        elif optimizer=='RProp': # Standard RProp
            max_iterations = options.get('max_iterations', 1000)
            def _optRho(rho, lam):
                return RProp(lambda x: _gradient(x,lam),rho,eps=eps,max_iterations=max_iterations)
        elif optimizer=='iRProp':
            max_iterations = options.get('max_iterations', 100)
            def _optRho(rho, lam):
                return iRProp(lambda x: _gradient(x,lam), lambda x: _bound(x,lam)[0], rho,\
                        eps=eps, max_iterations=max_iterations)
        #else: # Alternate - removed for now
        #    def _optRho(lam, rho):
        #        b        = _bound(rho, lam)
        #        bp       = b+2*eps
        #        while abs(b-bp) > eps:
        #            bp = b
        #            tndr_list = np.average(tandem_risks, weights=rho, axis=0)
        #            nrho = np.zeros(m)
        #            for i in range(m):
        #                nrho[i] = pi[i]*exp(-lam*n2*tndr_list[i])
        #            nrho /= np.sum(nrho)
        #            b = _bound(nrho, lam)
        #            if b > bp:
        #                b = bp
        #                break
        #            rho = nrho
        #        return rho

        b, lam   = _bound(rho)
        bp       = b+1
        while abs(b-bp) > eps:
            bp = b
            # Optimize rho
            nrho = _optRho(rho, lam)
            b, nlam = _bound(nrho)
            if b > bp:
                b = bp
                break
            rho, lam = nrho, nlam
        
        return (min(1.0,4*b), softmax(rho), lam)

# Optimize DIS
# options = {'optimizer':<opt>, 'max_iterations':<iter>, 'eps':<eps>, 'learning_rate':<lr>}
#
# Default for opt is iRProp
def optimizeDIS(gibbs_risks, disagreements, n, n2u, delta=0.05, options=None):
    options = dict() if options is None else options
    optimizer = options.get('optimizer', 'iRProp')
    
    if optimizer not in ['CMA', 'GD', 'RProp', 'iRProp']: # 'Alternate' removed for now
        warn('optimizeDIS: unknown optimizer: \''+optimizer+'\', using GD')
        optimizer = 'GD'
    
    m = gibbs_risks.shape[0]
    pi  = uniform_distribution(m)
    rho = uniform_distribution(m) 

    def _gr(rho): # Compute Gibbs risk from gibbs_risk list and rho
        return np.average(gibbs_risks, weights=rho)
    def _dis(rho): # Compute disagreement from disagreements matrix and rho
        return np.average(np.average(disagreements, weights=rho, axis=0), weights=rho)
    def _optLam(rho): # Compute optimal lambda for rho
        rho = softmax(rho)
        gr  = _gr(rho)
        KL  = kl(rho,pi)
        return 2.0 / (sqrt((2.0*n*gr)/(KL+log(4.0*sqrt(n)/delta)) + 1) + 1)
    def _optGam(rho): # Compute optimal gamma for rho
        rho = softmax(rho)
        dis = _dis(rho)
        if n2u*dis < 10**-9:
            return 2.0
        KL    = kl(rho,pi)
        return min(2.0, sqrt( (4.0*KL+log(16.0*n2u/delta**2)) / (n2u*dis) ))
    def _bound(rho, lam=None, gam=None): # Compute bound (and optimal lam and gam if None)
        rho = softmax(rho)
        gr  = _gr(rho)
        dis = _dis(rho)
        KL  = kl(rho,pi)
        if lam is None:
            lam = 2.0 / (sqrt((2.0*n*gr)/(KL+log(4.0*sqrt(n)/delta)) + 1) + 1)
        if gam is None:
            if n2u*dis < 10**-9:
                gam = 2.0
            else:
                gam = min(2.0, sqrt( (4.0*KL+log(16.0*n2u/delta**2)) / (n2u*dis) ))

        ub_gr  = gr/(1.0 - lam/2.0) + (KL+log(4.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
        lb_dis = (1.0-gam/2.0)*dis - (2*KL+log(4.0*sqrt(n2u)/delta))/(gam*n2u)
        bound  = 4*ub_gr - 2*lb_dis
        return (bound, lam, gam)

    if optimizer == 'CMA': # Optimize with CMA
        import cma
        def _func(r):
            return _bound(r)[0]
        sol = cma.fmin(_func, rho, 0.1, {'verbose': -9})
        b, lam, gam = _bound(sol[0])
        return (min(1.0,b), softmax(sol[0]), lam, gam)
    else: # 1st order optimizers
        def _gradient(rho, lam, gam):
            a = 1.0/(1.0-lam/2.0)
            b = 1-gam/2.0
            c = 1.0/(lam*(1.0-lam/2.0)*n) + 1.0/(gam*n2u)
            
            Srho = softmax(rho)
            # D_jS_i = S_i(1[i==j]-S_j)
            Smat = -np.outer(Srho, Srho)
            np.fill_diagonal(Smat, np.multiply(Srho,1.0-Srho))

            return 2*np.dot(a*gibbs_risks-b*np.dot(disagreements,Srho)+c*(1+np.log(Srho/pi)), Smat)
        
        max_iterations = options.get('max_iterations', None)
        eps = options.get('eps', 10**-9)
        
        if optimizer=='GD':
            def _optRho(rho, lam, gam):
                lr = options.get('learning_rate', 1)
                return GD(lambda x: _gradient(x,lam,gam), lambda x: _bound(x,lam,gam)[0], rho,\
                        max_iterations=max_iterations, lr=lr, eps=eps)
        elif optimizer=='RProp':
            max_iterations = options.get('max_iterations', 1000)
            def _optRho(rho, lam, gam):
                return RProp(lambda x: _gradient(x,lam,gam),rho,eps=eps,max_iterations=max_iterations)
        elif optimizer=='iRProp':
            max_iterations = options.get('max_iterations', 100)
            def _optRho(rho, lam, gam):
                return iRProp(lambda x: _gradient(x,lam,gam), lambda x: _bound(x,lam,gam)[0], rho,\
                        eps=eps,max_iterations=max_iterations)
        #else:
        #    def _optRho(lam, gam, rho):
        #        a = 1.0/(1.0-lam/2.0)
        #        b = 1-lam/2.0
        #        c = 1.0/(lam*(1.0-lam/2.0)*n) + 1.0/(gam*un)
        #        bound = _bound(lam, gam, rho)
        #        bprev = bound+2*eps
        #        while abs(bound-bprev) > eps:
        #            bprev = bound
        #            emp_dis_list = np.average(emp_dis_mat, weights=rho, axis=0)
        #            nrho = np.zeros(rho.shape[0])
        #            for i in range(rho.shape[0]):
        #                nrho[i] = pi[i]*exp(-(a/c)*emp_risk_list[i]+2.0*(b/c)*emp_dis_list[i])
        #            nrho /= np.sum(nrho)
        #            bound = _bound(lam, gam, nrho)
        #            if bound > bprev:
        #                break
        #            rho = nrho
        #        return rho

        rho = uniform_distribution(m)
        b, lam, gam  = _bound(rho)
        bp = b+1
        while abs(b-bp) > eps:
            bp = b
            # Optimize rho
            nrho = _optRho(rho,lam,gam)
            # Optimize lam + gam
            b, nlam, ngam = _bound(nrho)
            if b > bp:
                b = bp
                break
            rho, lam, gam = nrho, nlam, ngam
        
        return (min(1.0,b), softmax(rho), lam, gam)
