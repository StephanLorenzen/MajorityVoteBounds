#
# Implementation of the lambda bound and optimization procedure.
#
# Based on paper:
# [Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin.
#  A strongly quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017] 
#
import numpy as np

from math import ceil, log, sqrt, exp
from ..util import kl, uniform_distribution

# Compute PAC-Bayes-Lambda-bound:
def lamb(emp_risk, n, KL, delta=0.05):
    n = float(n)

    lamb = 2.0 / (sqrt((2.0*n*emp_risk)/(KL+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    bound = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)


# Optimize PAC-Bayes-Lambda-bound:
def optimizeLamb(emp_risks, n, delta=0.05, eps=10**-9, abc_pi=None):
    m = len(emp_risks)
    n = float(n)
    pi  = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    rho = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    KL = kl(rho,pi)

    lamb = 1.0
    emp_risk = np.average(emp_risks, weights=rho)

    upd = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
    bound = upd+2*eps

    while bound-upd > eps:
        bound = upd
        lamb = 2.0 / (sqrt((2.0*n*emp_risk)/(KL+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
        for h in range(m):
            rho[h] = pi[h]*exp(-lamb*n*emp_risks[h])
        rho /= np.sum(rho)

        emp_risk = np.average(emp_risks, weights=rho)
        KL = kl(rho,pi)

        upd = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n) 
    return bound, rho, lamb
