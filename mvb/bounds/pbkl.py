#-*- coding:utf-8 -*-
""" Computes the PBkl bound.
Implementation is modified version of code from paper:

Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

The PBkl function below is taken from the file pac_bound_0.py
The original documentation for the function is preserved.

Modifications:
- Changed name of function (pac_bound_zero -> PBkl)
- Fixed imports

Original documentation:
---
pac_bound_zero(...) function.
This file can be imported in your python project or executed as a command-line script.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/
"""

from .tools import validate_inputs, xi, solve_kl_sup
from math import log, sqrt

def PBkl(empirical_gibbs_risk, m, KLQP, delta=0.05):
    """ PAC Bound ZERO of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    multiplying by two an upper bound on the Gibbs risk

    empirical_gibbs_risk : Gibbs risk on the training set
    m : number of training examples
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    # Don't validate - gibbs_risk may be > 0.5 in non-binary case 
    #if not validate_inputs(empirical_gibbs_risk, None, m, KLQP, delta): return 1.0

    xi_m = 2*sqrt(m)
    right_hand_side = ( KLQP + log( xi_m / delta ) ) / m
    sup_R = min(1.0, solve_kl_sup(empirical_gibbs_risk, right_hand_side))

    return 2 * sup_R

