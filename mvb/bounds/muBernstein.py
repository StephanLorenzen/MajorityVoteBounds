#
# Implements the MU Bernstein bound.
#
import numpy as np
from math import log, sqrt, exp, e, ceil, nan
from .tools import solve_kl_sup, solve_kl_inf
from ..util import warn, kl, uniform_distribution, random_distribution, softmax, GD, RProp, iRProp 

# Implementation of MUBernstein
def MUBernstein(mutandem_risk, vartandem_risk, n2, KL, mu, delta=0.05):

    #We first compute a bound over the variance from Corollary 17
    varMuBound, _ = varMUBernstein(vartandem_risk, n2, KL, mu, delta1= delta/2.)

    #We plug the bound over variance to compute a bound over the muTandem loss following Corollary 20.
    muTandemBound, _ = muBernstein(mutandem_risk,varMuBound, n2, KL, mu, delta2= delta/2.)

    return min(1.0, muTandemBound/((0.5-mu)**2))


#Corollary 20 : \label{cor:pac-bayes-bernstein_grid}
def muBernstein(mutandem_risk, varMuBound, n2, KL, mu, c2=1.0, delta2=0.05, unionBound = False):

    if unionBound:
        nu2 = sqrt((e-2)*n2/(4*log(1/delta2)))
        nu2 = ceil(log(nu2)/log(c2))
    else:
        nu2 = 1
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)
    
    # E[varMuBound]<=Kmu^2/4
    varMuBound = min(varMuBound, Kmu**2/4)
    
    # From the proof of Collorary 20.
    bprime=c2*(2*KL + log(nu2) -log(delta2))/n2
    a=(e-2)*varMuBound
    gammastar = sqrt(bprime / a)
    
    # The range of Gamma^*
    gam_lb = sqrt(4*log(1/delta2)/(n2*(e-2)))/Kmu
    gam_ub = 1/Kmu
    
    if gammastar > gam_ub :
        gammastar = gam_ub
    elif gammastar < gam_lb:
        gammastar = gam_lb
    else:
        gammastar = gammastar
    
    bound = mutandem_risk + gammastar * a + bprime / gammastar
    return min(bound , 1.0), gammastar


#Corollary 17 : \label{cor:bound_variance_grid}
def varMUBernstein(vartandem_risk, n2, KL, mu, c1=1.0, delta1=0.05, unionBound = False):

    if unionBound:
        nu1  = (n2-1)/log(1/delta1)
        nu1 = 0.5*sqrt(nu1+1)+0.5
        nu1 = ceil(log(nu1)/log(c1))
    else:
        nu1 = 1

    # From the proof of Collorary 17.
    a = vartandem_risk
    bprime = c1*(2*KL + log(nu1) - log(delta1))/(2*(n2-1))
    
    # range factor
    Kmu = max(1-mu, 1-2*mu)

    # From the proof of Collorary 17.
    tstar2 = sqrt(a/(Kmu**2 * bprime)+1)+1
    tstar2 = 1./tstar2
    
    # From the proof of Collorary 17.
    varMuBound = a / (1 - tstar2) + Kmu**2 * bprime / (tstar2 * (1 - tstar2))

    lambdstar = 2*(n2-1)*tstar2/n2

    return varMuBound, lambdstar

