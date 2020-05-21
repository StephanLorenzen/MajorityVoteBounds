""" Computes the MV bound from the new paper.
"""

import numpy as np
import numpy.linalg as LA
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl as compute_kl, uniform_distribution, random_distribution

def MV(emp_jerr, n, KL, delta=0.05):
    rhs  = ( 2.0*KL + log(2.0*sqrt(n)/delta) ) / n
    jerr = min(0.25, solve_kl_sup(emp_jerr, rhs))
    return 4*jerr

def MVu(emp_risk, emp_dis, n, nu, KL, delta=0.05):
    g_rhs = ( KL + log(4.0*sqrt(n)/delta) ) / n
    g_ub  = min(1.0, solve_kl_sup(emp_risk, g_rhs))
    
    d_rhs = ( 2.0*KL + log(4.0*sqrt(nu)/delta) ) / nu
    d_lb  = solve_kl_inf(emp_dis, d_rhs)
    return min(1.0, 4*g_ub - 2*d_lb)

def optimizeMV(emp_jerr_mat, n, delta=0.05, eps=10**-9, optimizer='GD'):
    if optimizer not in ['Alternate', 'CMA', 'Newton', 'GD', 'GD-RProp']:
        warn('optimizeMV: unknown optimizer: \''+optimizer+'\', using GD')
        optimizer = 'GD'
    m = emp_jerr_mat.shape[0]
    rho = uniform_distribution(m)
    pi  = uniform_distribution(m)

    def _jerr(rho):
        return np.average(np.average(emp_jerr_mat, weights=rho, axis=0), weights=rho)
    
    def _optLam(jerr, KL):
        return 2.0 / (sqrt((2.0*n*jerr)/(2.0*KL+log(2.0*sqrt(n)/delta)) + 1) + 1)

    def _bound(emp_jerr, kl, lam):
        return emp_jerr / (1.0 - lam/2.0) + (2.0*kl+log(2.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
   
    if optimizer=='CMA':
        import cma
        def _softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        def _func(r):
            r = _softmax(r)
            KL = compute_kl(r, pi)
            jerr = _jerr(r)
            lam = _optLam(jerr, KL)
            return _bound(jerr, KL, lam)
        sol = cma.fmin(_func, rho, 0.1, {'verbose': -9})
        rho = _softmax(sol[0])
        KL = compute_kl(rho, pi)
        jerr = _jerr(rho)
        lam = _optLam(jerr, KL)
        b = _bound(jerr, KL, lam)

        return (min(1.0,4*b), rho, lam)
    else:
        def _gradient(lam, rho):
            return 2*(np.dot(emp_jerr_mat,rho)+1.0/(lam*n)*(1+np.log((rho/pi).clip(10**-10))))
        def _hessian(lam, rho):
            return 2*(emp_jerr_mat+1.0/(lam*n)*np.diag(1.0/rho))
        if optimizer=='Newton':
            print("Using newton")
            rho = random_distribution(m)
            num_iterations = 10
            lr = 1
            def _optRho(lam, rho):
                b = _bound(_jerr(rho), compute_kl(rho,pi), lam)
                print("Optrho",b)
                for t in range(1, num_iterations):
                    bp = b
                    nrho = rho - lr*np.dot(LA.inv(_hessian(lam, rho)), _gradient(lam, rho))
                    nrho /= np.sum(nrho)
                    b = _bound(_jerr(nrho), compute_kl(nrho,pi), lam)
                    print(b)
                    if b > bp:
                        b = bp
                        break
                    rho = nrho
                return rho
        elif optimizer=='GD':
            num_iterations = 100
            lr = 0.01
            def _optRho(lam, rho):
                b = _bound(_jerr(rho), compute_kl(rho,pi), lam)
                for t in range(1, num_iterations):
                    bp = b
                    nrho = rho - lr*_gradient(lam, rho)
                    nrho /= np.sum(nrho)
                    b = _bound(_jerr(nrho), compute_kl(nrho,pi), lam)
                    if b > bp:
                        b = bp
                        break
                    rho = nrho
                return rho
        elif optimizer=='GD-RProp':
            num_iterations = 10
            step_init = 0.01
            step_min  = 10**-12
            step_max  = 1
            inc_fact  = 1.1
            dec_fact  = 0.5
            rho = random_distribution(m)
            def _optRho(lam, rho):
                #print("Opt -> ",rho)
                dw  = np.zeros((num_iterations,m))
                w   = np.zeros((num_iterations,m))
                w[0]= rho
                step_size = np.ones((m,)) * step_init
                v_max = np.ones((m,)) * step_max
                v_min = np.ones((m,)) * step_min
                for t in range(1, num_iterations):
                    dw[t] = _gradient(lam, w[t-1])
   
                    prod = np.multiply(dw[t],dw[t-1])
                    step_inc  = np.min(np.vstack((step_size*inc_fact,v_max)), axis=0)
                    step_dec  = np.max(np.vstack((step_size*dec_fact,v_min)), axis=0)
                    step_inc[prod<=0] = 0.0
                    step_dec[prod>=0] = 0.0
                    step_size[prod!=0] = 0.0
                    step_size += step_inc+step_dec
                    #import pdb;pdb.set_trace()
                    #step_size = np.multiply(np.min(np.vstack((step_size*inc_fact,v_max)),axis=0), np.where(prod>0))\
                    #        + np.multiply(np.max(np.vstack((step_size*dec_fact,v_min)),axis=0), np.where(prod<0)) 
                    #if dw[t] * dw[t - 1] > 0:
                    #    step_size = min(step_size * inc_fact, step_max)
                    #elif dw[t] * dw[t - 1] < 0:
                    #    step_size = max(step_size * dec_fact, step_min)
   
                    w[t] = w[t-1] - np.multiply(np.sign(dw[t]), step_size)
                    #w[t] = w[t].clip(10**-10)
                    #w[t] /= np.sum(w[t])
                    #print(w[t])
                    #import pdb; pdb.set_trace()
                    emp_jerr = _jerr(w[t])
                    KL       = compute_kl(w[t], pi)
                    print("  ", _bound(emp_jerr, KL, lam))
                #import pdb; pdb.set_trace()
                return w[-1] / np.sum(w[-1])
        else:
            def _optRho(lam, rho):
                emp_jerr = _jerr(rho)
                KL       = compute_kl(rho, pi)
                b        = _bound(emp_jerr, KL, lam)
                bp       = b+2*eps
                while abs(b-bp) > eps:
                    bp = b
                    emp_jerr_list = np.average(emp_jerr_mat, weights=rho, axis=0)
                    nrho = np.zeros(rho.shape[0])
                    for i in range(rho.shape[0]):
                        nrho[i] = pi[i]*exp(-lam*n*emp_jerr_list[i])
                    nrho /= np.sum(nrho)
                    emp_jerr = _jerr(nrho)
                    KL       = compute_kl(nrho, pi)
                    b = _bound(emp_jerr, KL, lam)
                    if b > bp:
                        b = bp
                        break
                    rho = nrho
                return rho

        emp_jerr = _jerr(rho)
        KL       = compute_kl(rho, pi)
        lam      = _optLam(emp_jerr, KL)
        b        = _bound(emp_jerr, KL, lam)
        bp       = b+2*eps
        while abs(b-bp) > eps:
            bp = b
            # Optimize rho
            rho = _optRho(lam, rho)
            #print(rho)
            # Optimize lam
            emp_jerr = _jerr(rho)
            KL       = compute_kl(rho, pi)
            lam      = _optLam(emp_jerr, KL)
            b        = _bound(emp_jerr, KL, lam)
            if b > bp:
                break
            #print(b)
        return (min(1.0,4*b), rho, lam)

def optimizeMVu(emp_risk_list, emp_dis_mat, n, un, delta=0.05, eps=10**-9, optimizer='GD'):
    if optimizer not in ['Alternate', 'CMA', 'Newton', 'GD']:
        warn('optimizeMV: unknown optimizer: \''+optimizer+'\', using GD')
        optimizer = 'GD'
    m = emp_risk_list.shape[0]
    pi  = uniform_distribution(m)
    
    def _erisk(rho):
        return np.average(emp_risk_list, weights=rho)
    def _jdis(rho):
        return np.average(np.average(emp_dis_mat, weights=rho, axis=0), weights=rho)
    def _optLam(emp_risk, KL):
        return 2.0 / (sqrt((2.0*n*emp_risk)/(KL+log(4.0*sqrt(n)/delta)) + 1) + 1)
    def _optGam(emp_dis, KL):
        if un*emp_dis < 10**-9:
            return 2.0
        return min(2.0, sqrt( (4.0*KL+log(16.0*un/delta**2)) / (un*emp_dis) ))
    def _bound(lam, gam, rho):
        emp_risk = _erisk(rho)
        emp_dis  = _jdis(rho)
        KL = compute_kl(rho, pi)

        grisk = emp_risk/(1.0 - lam/2.0) + (KL+log(4.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
        jrisk = (1.0-gam/2.0)*emp_dis - (2*KL+log(4.0*sqrt(un)/delta))/(gam*un)
   
        return 4*grisk - 2*jrisk

    rho = uniform_distribution(m)
    if optimizer == 'CMA':
        import cma
        def _bound_cma(emp_risk, emp_dis, KL, lam, gam, rho):
            grisk = emp_risk/(1.0 - lam/2.0) + (KL+log(4.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
            jrisk = (1.0-gam/2.0)*emp_dis - (2*KL+log(4.0*sqrt(un)/delta))/(gam*un)
            return 4*grisk - 2*jrisk
        def _softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        def _func(r):
            r = _softmax(r)
            KL = compute_kl(r, pi)
            erisk = _erisk(r)
            edis  = _jdis(r)
            lam = _optLam(erisk, KL)
            gam = _optGam(edis, KL)
            return _bound_cma(erisk, edis, KL, lam, gam, r)
        sol = cma.fmin(_func, rho, 0.1, {'verbose': -9})
        rho = _softmax(sol[0])
        KL = compute_kl(rho, pi)
        erisk = _erisk(rho)
        edis  = _jdis(rho)
        lam = _optLam(erisk, KL)
        gam = _optGam(edis, KL)
        b = _bound(lam, gam, rho)
        return (min(1.0,b), rho, lam, gam)
    else:
        def _gradient(lam, gam, rho):
            a = 1.0/(1.0-lam/2.0)
            b = 1-lam/2.0
            c = 1.0/(lam*(1.0-lam/2.0)*n) + 1.0/(gam*un)
            return 2*(a*emp_risk_list-b*np.dot(emp_dis_mat,rho)+c*(1+np.log((rho/pi).clip(10**-10))))
        def _hessian(lam, gam, rho):
            b = 1-lam/2.0
            c = 1.0/(lam*(1.0-lam/2.0)*n) + 1.0/(gam*un)
            return 2*(-b*emp_dis_mat+c*np.diag(1.0/rho))
        if optimizer=='Newton':
            print("Using newton")
            rho = random_distribution(m)
            num_iterations = 10
            lr = 1
            def _optRho(lam, gam, rho):
                b = _bound(lam, gam, rho)
                print("Optrho",b)
                for t in range(1, num_iterations):
                    bp = b
                    nrho = rho - lr*np.dot(LA.inv(_hessian(lam, gam, rho)), _gradient(lam, gam, rho))
                    nrho /= np.sum(nrho)
                    b = _bound(lam, gam, rho)
                    print(b)
                    if b > bp:
                        b = bp
                        break
                    rho = nrho
                return rho
        elif optimizer=='GD':
            num_iterations = 100
            lr = 0.01
            def _optRho(lam, gam, rho):
                b = _bound(lam, gam, rho)
                for t in range(1, num_iterations):
                    bp = b
                    nrho = rho - lr*_gradient(lam, gam, rho)
                    nrho /= np.sum(nrho)
                    b = _bound(lam, gam, nrho)
                    if b > bp:
                        b = bp
                        break
                    rho = nrho
                return rho
        elif optimize=='GD-RProp':
            pass
        else:
            def _optRho(lam, gam, rho):
                a = 1.0/(1.0-lam/2.0)
                b = 1-lam/2.0
                c = 1.0/(lam*(1.0-lam/2.0)*n) + 1.0/(gam*un)
                bound = _bound(lam, gam, rho)
                bprev = bound+2*eps
                while abs(bound-bprev) > eps:
                    bprev = bound
                    emp_dis_list = np.average(emp_dis_mat, weights=rho, axis=0)
                    nrho = np.zeros(rho.shape[0])
                    for i in range(rho.shape[0]):
                        nrho[i] = pi[i]*exp(-(a/c)*emp_risk_list[i]+2.0*(b/c)*emp_dis_list[i])
                    nrho /= np.sum(nrho)
                    bound = _bound(lam, gam, nrho)
                    if bound > bprev:
                        break
                    rho = nrho
                return rho

        emp_risk = _erisk(rho)
        emp_dis  = _jdis(rho)
        KL  = compute_kl(rho, pi)
        lam = _optLam(emp_risk, KL)
        gam = _optGam(emp_dis, KL)
        b  = _bound(lam, gam, rho)
        bp = b+2*eps
        while abs(b-bp) > eps:
            bp = b
            # Optimize rho
            rho = _optRho(lam,gam,rho)
            # Optimize lam + gam
            emp_risk = _erisk(rho)
            emp_dis  = _jdis(rho)
            KL       = compute_kl(rho, pi)
            lam = _optLam(emp_risk, KL)
            gam = _optGam(emp_dis, KL)
            b = _bound(lam, gam, rho)
            if b > bp:
                break
         
        return (min(1.0,b), rho, lam, gam)
