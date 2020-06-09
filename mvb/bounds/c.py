#-*- coding:utf-8 -*-
""" Computes the PBkl bound.
Implementation is modified version of code from paper:

Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

The C1 and C2 functions below are taken from file pac_bound_1.py and pac_bound_2.py resp.
The original documentation for each function is preserved, with small modifications:

Modifications:
- Changed name name of function (pac_bound_one -> C1)
- Changed name name of function (pac_bound_two -> C2)
- Changed input of C2 (from empirical_gibbs_risk to empirical_joint_error
- Fixed imports

Original documentation:
--- [pac_bound_1.py]
pac_bound_one(...) function.
This file can be imported in your python project or executed as a command-line script.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/

--- [pac_bound_2.py]
pac_bound_two(...) function.
This file can be imported in your python project or executed as a command-line script.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/
"""

from .tools import validate_inputs, xi, solve_kl_inf, solve_kl_sup, c_bound_third_form, maximize_c_bound_under_constraints
from math import log, sqrt

def C1(empirical_gibbs_risk, empirical_disagreement, m, md, KLQP, delta=0.05):
    """ PAC Bound ONE of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    using the C-Bound on an upper bound on the Gibbs risk
    and a lower bound on the expected disagreement

    empirical_gibbs_risk : Gibbs risk on the training set
    empirical_disagreement : Expected disagreement on the training set
    m  : minimum number of training samples used for any voter
    md : minimum number of training samples overlapping between any two voters
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    # Removed check, as a) no longer using empirical_gibbs_risk for input and
    # b), the check may fail in the OOB setting (while not actually being invalid)
    # if not validate_inputs(empirical_gibbs_risk, empirical_disagreement, m, KLQP, delta): return 1.0
    if empirical_gibbs_risk > 0.5:
        return 1.0

    xi_m = xi(m)
    right_hand_side = ( KLQP + log( 2 * xi_m / delta ) ) / m
    sup_R = min(0.5, solve_kl_sup(empirical_gibbs_risk, right_hand_side))
    xi_md = xi(md)
    right_hand_side = ( 2*KLQP + log( 2 * xi_md / delta ) ) / md
    inf_D = solve_kl_inf(empirical_disagreement, right_hand_side)
    return c_bound_third_form(sup_R, inf_D)

def C2(empirical_joint_error, empirical_disagreement, md, KLQP, delta=0.05):
    """ PAC Bound TWO of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    using the C-Bound. To do so, we bound *simultaneously*
    the disagreement and the joint error.

    empirical_gibbs_risk : Gibbs risk on the training set
    empirical_disagreement : Expected disagreement on the training set
    md : minimum number of training samples overlapping between any two voters
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    # Removed check, as a) no longer using empirical_gibbs_risk for input and
    # b), the check may fail in the OOB setting (while not actually being invalid)
    #if not validate_inputs(empirical_gibbs_risk, empirical_disagreement, m, KLQP, delta): return 1.0

    xi_md = xi(md)
    right_hand_side  = (2*KLQP + log( (xi_md+md)/delta ) ) / md

    return maximize_c_bound_under_constraints(empirical_disagreement, empirical_joint_error, right_hand_side )

def C3(empirical_gibbs_risk, empirical_gibbs_disagreement, m, md, delta=0.05):
    """ PAC Bound THREE. of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    empirical_gibbs_risk : Gibbs risk on the training set
    empirical_disagreement : Expected disagreement on the training set
    m  : minimum number of training samples used for any voter
    md : minimum number of training samples overlapping between any two voters
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    xi_m  = float(xi(m))
    xi_md = float(xi(md))
    r = min(0.5, empirical_gibbs_risk+sqrt(1.0/(2.0*m)*log(xi_m/(delta/2.0))))
    d = max(0.0, empirical_gibbs_disagreement - sqrt(1.0/(2.0*md)*log(xi_md/(delta/2.0))))
    return 1.0-(1.0-2.0*r)**2/(1.0-2.0*d)

