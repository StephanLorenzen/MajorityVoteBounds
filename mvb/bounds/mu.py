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

    rhs_tr = ( 2.0*KL + log(4.0*sqrt(n2)/delta) ) / n2
    # UpperBound_TandemRisk
    ub_tr  = solve_kl_sup(tandem_risk, rhs_tr)
    
    rhs_gr = ( KL + log(4.0*sqrt(n)/delta) ) / n
    # LowerBound_GibbsRisk
    lb_gr  = solve_kl_inf(gibbs_risk, rhs_gr)
   
    # Compute for mu_grid
    bnd = 0.0
    for mu in mu_grid:
        bnd += 4*(ub_tr - 2*mu*lb_gr + mu**2)
    return min(1.0, bnd)

