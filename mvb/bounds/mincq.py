# Implementation of the MinCq algorithm of Germain et al. Unused.
import numpy as np
from cvxopt import matrix, solvers
from cvxpy import ECOS

def MinCq(risks, disM, mu=0.1):
    e = risks.shape[0]
    m = np.ones(e)-2.0*risks
    M = np.ones((e,e)) - 2.0*disM
    rho_1st   = MinCqRaw(m, M, mu)
    rho_2nd   = np.array([1.0/float(e)]*e)-rho_1st
    return np.append(rho_1st, rho_2nd)

def MinCqRaw(m, M, mu):
    n = m.shape[0]

    A = matrix(m, (1,n))
    q = matrix((-1.0)*(np.sum(M, axis=0)/float(n))) # a in paper
    P = matrix(2.0*M)
    q *= 10**(-5)
    P *= 10**(-5)

    G = np.vstack([-np.eye(n),np.eye(n)])
    h = np.vstack([np.zeros((n,1)), np.repeat(1.0/float(n),n).reshape(n,1)])
    G = matrix(G)
    h = matrix(h)

    b = mu/2.0 + 1.0/(2.0*n) * np.sum(m)
    b = matrix(b)
    
    solvers.options['feastol']=1e-1
    
    sol = solvers.qp(P,q,G,h,A,b) 

    return np.array(sol['x']).flatten()
