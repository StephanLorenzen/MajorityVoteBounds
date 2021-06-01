#-*- coding:utf-8 -*-
""" Tools for bound computation.
Implementation is taken from paper:

Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

The file is renamed from the file pac_bound_tools.py
The original documentation for each function is preserved.

Original documentation:
---
Various functions imported by bound computation files pac_bound_{0,1,1p,2,2p}.py.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/
"""

from math import log, sqrt
import numpy as np
from scipy.special import gammaln
from scipy import optimize

def validate_inputs(empirical_gibbs_risk, empirical_disagreement=None, m=None, KLQP=None, delta=0.05):
    """
    This utility function validates if entry parameters are plausible when computing
    PAC-Bayesian bounds.
    """
    is_valid = [True]
    def handle_error(msg):
        print('INVALID INPUT: ' + msg)
        is_valid[0] = False

    if empirical_gibbs_risk < 0.0 or empirical_gibbs_risk >= 0.5:
        handle_error( 'empirical_gibbs_risk must lies in [0.0,0.5)' )
    if empirical_disagreement is not None:
        if empirical_disagreement < 0.0 or empirical_disagreement >= 0.5:
            handle_error( 'empirical_disagreement must lies in [0.0,0.5)' )
        if empirical_disagreement > 2*empirical_gibbs_risk*(1.0-empirical_gibbs_risk):
            handle_error( 'invalid variance, i.e., empirical_disagreement > 2*empirical_gibbs_risk*(1.0-empirical_gibbs_risk)' )
    if m is not None and m <=0:
        handle_error( 'm must be strictly positive.' )
    if KLQP is not None and KLQP < 0.0:
        handle_error( 'KLQP must be positive.' )
    if delta <= 0.0 or delta >= 0.5:
        handle_error( 'delta must lies in (0.0, 1.0)' )

    return is_valid[0]


def c_bound_third_form(gibbs_risk, disagreement):
    """
    Compute the C-bound according to the Gibbs risk and
    the expected disagreement of a weighted set of voters.
    """
    return 1.0 - (1.0 - 2*gibbs_risk)**2 / (1.0 - 2*disagreement)


def xi(m):
    """
    Compute complexity term xi(m) of PAC-Bayesian bounds,
    where m is the number of training examples.
    """
    k = np.arange(1, m, 1.0)
    k_over_m = k/float(m)
    return 2.0 + np.sum( np.exp( gammaln(m+1.0) - gammaln(k+1.0) - gammaln(m-k+1.0) + k*np.log(k_over_m) + (m-k)*np.log(1.0-k_over_m) ) )


def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([ q*log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])


def KL_trinomial(q1, q2, p1, p2):
    """
    Compute the KL-divergence between two multinomial distributions (Q and P)
    with three possible events, where Q=(q1,q2,1-q1-q2), P=(p1,p2,1-p1-p2).
    """
    return KL([q1, q2, 1.-q1-q2], [p1, p2,  1.-p1-p2])


def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0-1e-9) <= 0.0:
        return 1.0-1e-9
    else:
        return optimize.brentq(f, q, 1.0-1e-11)


def solve_kl_inf(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x < q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1e-9) <= 0.0:
        return 1e-9
    else:
        return optimize.brentq(f, 1e-11, q)

def Lambert(c, branch):
    """
    Implement the Lambert W function, i.e., the solution of xe^x=c, by Newton's method.
    Requirement : c\geq -1/e
    branch W0 : the solution x\geq -1
    branch W-1 : the solution x\leq -1
    """
    eps = 1e-9
    
    # fx(x, c) = xe^x-c
    def _fx(x, c):
        return x*np.exp(x)-c
        
    # fx_prime(x) = (x+1)e^x, first derivative of _fx
    def _fx_prime(x):
        return (x+1)*np.exp(x)
    
    if branch == 'W0':
        x_old = 0 # initial guess for the principle branch
    elif branch == 'W-1':
        x_old = -3  # initial guess for the -1 branch
    else:
        Warning('No such branch')
        
    diff = abs(_fx(x_old, c))
    while(diff > eps):
        x_new = x_old - _fx(x_old, c)/_fx_prime(x_old)
        x_old = x_new
        diff = _fx(x_old, c)
    
    return x_new

def maximize_c_bound_under_constraints(empirical_disagreement, empirical_joint_error, right_hand_side, sup_joint_error=0.5):
    """
    maximize F(d,e) such that:
        kl( empirical_disagreement, empirical_joint_error || d,e ) <= right_hand_side
        2*e + d < 1 (i.e., the Gibbs risk is less than 1/2)
        d <= 2*[sqrt(e)-e] (i.e., the margin variance is positive)
        e <= sup_joint_error (default: 0.5; used by PAC-Bound 2 prime)
    """

    # Objective function
    objective_fct = lambda e,d: -1 * c_bound_third_form(e+d/2, d)

    # Domain constraint given by the KL-divergence
    domain_fct = lambda e,d: KL_trinomial(empirical_joint_error, empirical_disagreement, e, d) - right_hand_side

    # If the constraint 2*e + d < 1 crosses the domain, the bound is trivial
    if empirical_disagreement > 0.0:
        if domain_fct( (1.0-empirical_disagreement)/2, empirical_disagreement) < 0.0:
            return 1.0

    # Find max value of joint error inside the domain
    find_d_minimizing_KL_given_e = lambda e: (e-1.0)*empirical_disagreement/(empirical_joint_error-1.0)
    minimize_domain_fct_given_e = lambda e: domain_fct( e, find_d_minimizing_KL_given_e(e) )
    e_max = optimize.brentq(minimize_domain_fct_given_e, empirical_joint_error, .5)
    e_max = min( e_max, sup_joint_error)

    # Given a fixed value of joint error, maximize the objective under the domain constraints
    def minimize_obj_given_e(_e):
        objective_fct_fixed_e = lambda d: objective_fct(_e, d)
        domain_fct_fixed_e = lambda d: domain_fct(_e, d)

        d_min = 0.
        d_max = 2 * (sqrt(_e) - _e)
        d_inside_domain = find_d_minimizing_KL_given_e(_e)
        if empirical_disagreement > 0. and False: # Hack
            d_min = optimize.brentq(domain_fct_fixed_e, 1e-9, d_inside_domain)
        if domain_fct_fixed_e(d_max) > 0.:
            d_max = optimize.brentq(domain_fct_fixed_e, d_inside_domain, d_max)
        
        if(d_max<5*10**(-5)): # Small hack
            optimal_d = 0
        else:
            optimal_d = optimize.fminbound( objective_fct_fixed_e, d_min, d_max)
        return objective_fct(_e, optimal_d)

    # Solve the optimization problem!
    obj_value = optimize.fminbound( minimize_obj_given_e, empirical_joint_error, e_max, full_output=True)[1]
    return -1 * obj_value

