""" Computes the MV bound from the new paper.
"""

import numpy as np
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import kl as compute_kl, uniform_distribution

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

def optimizeMV(emp_jerr_mat, n, delta=0.05, eps=10**-9, use_cma=True):
    m = emp_jerr_mat.shape[0]
    rho = uniform_distribution(m)
    pi  = uniform_distribution(m)
    if use_cma:
        import cma

    def _jerr(rho):
        return np.average(np.average(emp_jerr_mat, weights=rho, axis=0), weights=rho)
    
    def _optLam(jerr, KL):
        return 2.0 / (sqrt((2.0*n*jerr)/(2.0*KL+log(2.0*sqrt(n)/delta)) + 1) + 1)
    def _optRho(lam, rho):
        emp_jerr_list = np.average(emp_jerr_mat, weights=rho, axis=0)
        
        nrho = np.zeros(rho.shape[0])
        for i in range(rho.shape[0]):
            nrho[i] = pi[i]*exp(-lam*n*emp_jerr_list[i])
        
        return nrho / np.sum(nrho)

    def _bound(emp_jerr, kl, lam):
        return emp_jerr / (1.0 - lam/2.0) + (2.0*kl+log(2.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
   
    if use_cma:
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
        emp_jerr = _jerr(rho)
        KL       = compute_kl(rho, pi)
        lam      = _optLam(emp_jerr, KL)
        b        = _bound(emp_jerr, KL, lam)
        bp       = b+2*eps
        while abs(b-bp) > eps:
            bp = b
            # Optimize rho
            bpr = b+2*eps
            while abs(b-bpr) > eps:
                bpr = b
                nrho = _optRho(lam, rho)
                emp_jerr = _jerr(nrho)
                KL       = compute_kl(nrho, pi)
                b = _bound(emp_jerr, KL, lam)
                if b > bpr:
                    # Did not improve
                    b = bpr
                    break
                # Did improve - re-iterate
                rho = nrho
        # Optimize lam
        emp_jerr = _jerr(rho)
        KL       = compute_kl(rho, pi)
        lam = _optLam(emp_jerr, KL)
        b = _bound(emp_jerr, KL, lam)

        return (min(1.0,4*b), rho, lam)

def optimizeMVu(emp_risk_list, emp_dis_mat, n, un, delta=0.05, eps=10**-9, use_cma=True):
    m = emp_risk_list.shape[0]
    pi  = uniform_distribution(m)
    if use_cma:
        import cma
    
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
    def _optRho(lam, gam, rho):
        a = 2.0/(1.0-lam/2.0)
        b = 1-lam/2.0
        c = 2.0/(lam*(1.0-lam/2.0)*n) + 2.0/(gam*un)

        emp_dis_list = np.average(emp_dis_mat, weights=rho, axis=0)
        nrho = np.zeros(rho.shape[0])
        for i in range(rho.shape[0]):
            nrho[i] = pi[i]*exp(-(a/c)*emp_risk_list[i]+2.0*(b/c)*emp_dis_list[i])
        
        return nrho / np.sum(nrho)

    def _bound(lam, gam, rho):
        emp_risk = _erisk(rho)
        emp_dis  = _jdis(rho)
        KL = compute_kl(rho, pi)

        grisk = emp_risk/(1.0 - lam/2.0) + (KL+log(4.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
        jrisk = (1.0-gam/2.0)*emp_dis - (2*KL+log(4.0*sqrt(un)/delta))/(gam*un)
   
        return 4*grisk - 2*jrisk
    def _bound_cma(emp_risk, emp_dis, KL, lam, gam, rho):
        grisk = emp_risk/(1.0 - lam/2.0) + (KL+log(4.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
        jrisk = (1.0-gam/2.0)*emp_dis - (2*KL+log(4.0*sqrt(un)/delta))/(gam*un)
   
        return 4*grisk - 2*jrisk

    rho = uniform_distribution(m)
    if use_cma:
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
            bpr = b+2*eps
            while abs(b-bpr) > eps:
                bpr = b
                nrho = _optRho(lam, gam, rho)
                b = _bound(lam, gam, nrho)
                if b > bpr:
                    # Did not improve
                    b = bpr
                    break
                # Did improve - re-iterate
                rho = nrho
            # Optimize lam + gam
            emp_risk = _erisk(rho)
            emp_dis  = _jdis(rho)
            KL       = compute_kl(rho, pi)
            lam = _optLam(emp_risk, KL)
            gam = _optGam(emp_dis, KL)
            b = _bound(lam, gam, rho)
    
        return (min(1.0,b), rho, lam, gam)
