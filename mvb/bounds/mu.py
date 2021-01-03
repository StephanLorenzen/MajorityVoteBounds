#
# Implements the MU bound.
#
import numpy as np
from math import log, sqrt, exp
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 

# Implementation of MU
def MU(tandem_risk, gibbs_risk, n, n2, KL, mu_grid=[0.0], delta=0.05):
    if gibbs_risk > 0.5:
        return 1.0
    if len(mu_grid)<1:
        return 1.0

    # Union bound over K = len(mu_grid) -> delta = delta/K
    K = len(mu_grid)
    delta /= K

    # UpperBound_TandemRisk
    rhs_tr = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
    ub_tr  = solve_kl_sup(tandem_risk, rhs_tr)
    
    # LowerBound_GibbsRisk
    rhs_gr = ( KL + log(4.0*sqrt(n)/delta) ) / n
    lb_gr  = solve_kl_inf(gibbs_risk, rhs_gr)
   
    # Compute K bounds
    bnds = []
    for mu in mu_grid:
        bnds.append((ub_tr - 2*mu*lb_gr + mu**2)/(0.5-mu)**2)
    return min(1.0, min(bnds))

