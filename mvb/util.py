#
# Various utility functions
#
import numpy as np
import numpy.linalg as la
from math import log
from sklearn.metrics import accuracy_score

######################
# Error handling

def error(msg):
    raise Exception(msg)

def warn(msg):
    print(msg)



#######################
# Distributions and KL

def kl(rho,pi):
    m = pi.shape[0]
    assert(rho.shape[0]==m)
    kl = 0.0
    for h in range(m):
        kl += rho[h]*log(rho[h]/pi[h]) if rho[h]>10**-12 else 0
    return kl

def uniform_distribution(m):
    return (1.0/m) * np.ones((m,))

def random_distribution(m):
    dist = np.random.random(m)
    return dist/np.sum(dist)

def softmax(dist):
    dexp = np.exp(dist)
    return dexp / np.sum(dexp, axis=0)

######################
# Various stats

# Simply computes error
# preds.shape = (n,)
# targs.shape = (n,)
def risk(preds, targs):
    assert(preds.shape == targs.shape)
    return 1.0-accuracy_score(preds,targs)


# Computes the number of prediction errors for each classifier
# preds.shape = (m, n)
# targs.shape = (n,)
def risks(preds, targs):
    assert(len(preds.shape)==2 and len(targs.shape)==1)
    assert(preds.shape[1] == targs.shape[0])
    res = []
    for j in range(preds.shape[0]):
        res.append(np.sum(preds[j]!=targs))
    return np.array(res)

def disagreements(preds):
    m,n = preds.shape
    disagreements = np.zeros((m,m))
    for i in range(m):
        for j in range(i, m):
            dis = np.sum(preds[i]!=preds[j])
            disagreements[i,j] += dis
            if i != j:
                disagreements[j,i] += dis
    
    return disagreements

def tandem_risks(preds, targs):
    m,n = preds.shape
    tandem_risks = np.zeros((m,m))
    for i in range(m):
        for j in range(i, m):
            tand = np.sum(np.logical_and((preds[i]!=targs), (preds[j]!=targs)))
            tandem_risks[i,j] += tand
            if i != j:
                tandem_risks[j,i] += tand
    return tandem_risks

def mutandem_risks(preds, targs, mu):
    m,n = preds.shape
    mutandem_risks = np.zeros((m,m))
    mutandem_risks_P = np.zeros((m,m))
    mutandem_risks_M = np.zeros((m,m))
    musquaretandem_risks = np.zeros((m,m))
    mu_prime = mu**2 if mu>=0 else -mu*(1-mu)

    for i in range(m):
        for j in range(i, m):
            tand = np.sum(np.multiply((preds[i]!=targs)-mu, (preds[j]!=targs)-mu))
            tand_P = np.sum(np.max(0, np.multiply((preds[i]!=targs)-mu, (preds[j]!=targs)-mu)-mu_prime))
            tand_M = np.sum(np.max(0, mu_prime-np.multiply((preds[i]!=targs)-mu, (preds[j]!=targs)-mu)))
            squaretand = np.sum(np.square(np.multiply((preds[i] != targs) - mu, (preds[j] != targs) - mu)))
            mutandem_risks[i,j] += tand
            mutandem_risks_P[i,j] += tand_P
            mutandem_risks_M[i,j] += tand_M
            musquaretandem_risks[i,j] += squaretand
            if i != j:
                mutandem_risks[j,i] += tand
                mutandem_risks_P[j,i] += tand_P
                mutandem_risks_M[j,i] += tand_M
                musquaretandem_risks[j, i] += squaretand
    return mutandem_risks, mutandem_risks_P, mutandem_risks_M, musquaretandem_risks

#######################
# OOB 

def oob_estimate(rho, preds, targs):
    m = rho.shape[0]
    assert(len(preds)==m)
    n = targs.shape[0]
   
    Ps = [[] for _ in range(n)]
    Ws = [0.0]*n
    for i in range(m):
        i_idx, i_preds = preds[i]
        for idx, pred in zip(i_idx, i_preds):
            Ps[idx].append((rho[i],pred))
    
    results = np.zeros((n,))
    incl    = np.ones((n,))
    for i, pl in enumerate(Ps):
        if(len(pl)==0):
            # Elem used for training all. Exlude from result
            incl[i] = 0
            continue
        bins = dict()
        for r,vote in pl:
            vote = int(vote)
            if vote not in bins:
                bins[vote] = 0.0
            bins[vote] += r
        
        res   = -1
        mvote = 0
        for b, vote in bins.items():
            if vote > mvote:
                mvote = vote
                res = b
        results[i] = res
    return risk(results[np.where(incl)], targs[np.where(incl)])

def oob_risks(preds, targs):
    m     = len(preds)
    risks = np.zeros((m,))
    ns    = np.zeros((m,))
    for j, (M, P) in enumerate(preds):
        risks[j] = np.sum(P[M==1]!=targs[M==1])
        ns[j] = np.sum(M)
    return risks, ns

def oob_disagreements(preds):
    m = len(preds)
    disagreements = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            disagreements[i,j] = np.sum(P_i[M==1]!=P_j[M==1])
            n2[i,j] = np.sum(M)
            
            if i != j:
                disagreements[j,i] = disagreements[i,j]
                n2[j,i]            = n2[i,j]
    return disagreements, n2    

def oob_tandem_risks(preds, targs):
    m = len(preds)
    tandem_risks  = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            tandem_risks[i,j] = np.sum(np.logical_and(P_i[M==1]!=targs[M==1], P_j[M==1]!=targs[M==1]))
            n2[i,j] = np.sum(M)
            
            if i != j:
                tandem_risks[j,i] = tandem_risks[i,j]
                n2[j,i]           = n2[i,j]
    
    return tandem_risks, n2


def oob_mutandem_risks(preds, targs, mu):
    m = len(preds)
    mutandem_risks = np.zeros((m, m))
    mutandem_risks_P = np.zeros((m, m))
    mutandem_risks_M = np.zeros((m, m))
    musquaretandem_risks = np.zeros((m, m))
    n2 = np.zeros((m, m))
    mu_prime = mu**2 if mu>=0 else -mu*(1-mu)

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i, M_j)
            mutandem_risks[i, j] = np.sum(np.multiply((P_i[M == 1] != targs[M == 1]) - mu, (P_j[M == 1] != targs[M == 1])-mu))
            zero_vector = np.zeros(int(np.sum(M)))
            mutandem_risks_P[i, j] = np.sum(np.maximum(zero_vector, np.multiply((P_i[M == 1] != targs[M == 1]) - mu, (P_j[M == 1] != targs[M == 1])-mu)-mu_prime))
            mutandem_risks_M[i, j] = np.sum(np.maximum(zero_vector, mu_prime-np.multiply((P_i[M == 1] != targs[M == 1]) - mu, (P_j[M == 1] != targs[M == 1])-mu)))
            musquaretandem_risks[i, j] = np.sum(np.square(np.multiply((P_i[M == 1] != targs[M == 1]) - mu, (P_j[M == 1] != targs[M == 1])-mu)))
            n2[i, j] = np.sum(M)

            if i != j:
                mutandem_risks[j, i] = mutandem_risks[i, j]
                mutandem_risks_P[j, i] = mutandem_risks_P[i, j]
                mutandem_risks_M[j, i] = mutandem_risks_M[i, j]
                musquaretandem_risks[j, i] = musquaretandem_risks[i, j]
                n2[j, i] = n2[i, j]

    return mutandem_risks, mutandem_risks_P, mutandem_risks_M, musquaretandem_risks, n2


#######################
# Majority vote risk

def mv_risk(rho, preds, targs):
    mvPreds = mv_preds(rho, preds)
    return risk(mvPreds, targs) 

def mv_preds(rho, preds):
    m = rho.shape[0]
    preds = np.transpose(preds)
    assert(preds.shape[1] == m)
    n = preds.shape[0]

    tr = np.min(preds)
    preds -= tr

    results = np.zeros(n)
    for i,pl in enumerate(preds):
        results[i] = np.argmax(np.bincount(pl, weights=rho))
    return results+tr

##########################
# Optimizers

def GD(grad, func, x0, max_iterations=None, eps=10**-9, lr=0.1):
    max_iterations=10000 if max_iterations is None else max_iterations

    x  = x0
    fv = func(x)
    lr *= x0.shape[0]
    for t in range(1, max_iterations):
        x1  = x - lr*grad(x)
        fv1 = func(x1)
        if abs(fv-fv1) < eps or lr < eps:
            break
        if fv1 > fv:
            lr *= 0.5
            fv1 = fv
        else:
            x  = x1
            fv = fv1
    return x

def RProp(grad, x0,
        max_iterations=None,\
        eps=10**-9,\
        step_init=0.1,\
        step_min=10**-20,\
        step_max=10**5,\
        inc_fact=1.1,\
        dec_fact=0.5):
    max_iterations=10000 if max_iterations is None else max_iterations
    
    dim  = x0.shape[0]
    dw   = np.zeros((max_iterations, dim))
    w    = np.zeros((max_iterations, dim))
    w[0] = x0
    s_size = np.ones(dim) * step_init
    t = 1
    while t < max_iterations:
        dw[t] = grad(w[t-1])
        if la.norm(dw[t]) < eps:
            break
    
        # Update set size
        det = np.multiply(dw[t], dw[t-1])
        # Increase where det>0
        s_size[det>0] = s_size[det>0]*inc_fact
        # Decrease where det<0
        s_size[det<0] = s_size[det<0]*dec_fact
        # Upper/lower bound by min/max
        s_size        = s_size.clip(step_min, step_max)

        # Update w
        w[t] = w[t-1] - np.multiply(np.sign(dw[t]), s_size)
        t += 1
    return w[t-1]

# Implementation based on paper:
#
# [Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly
#  quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017]
#
def iRProp(grad, func, x0,
        max_iterations=None,\
        eps=10**-9,\
        step_init=0.1,\
        step_min=10^-20,\
        step_max=10**5,\
        inc_fact=1.1,\
        dec_fact=0.5):
    max_iterations=1000 if max_iterations is None else max_iterations
    
    n    = x0.shape[0]
    dx   = np.zeros((max_iterations, n))
    x    = np.zeros((max_iterations, n))
    x[0] = x0
    step = np.ones(n)*step_init
    
    fx   = np.ones(max_iterations)
    fx[0]= func(x[0])
    tb   = 0

    t = 1
    while t < max_iterations:
        delta = fx[t-1]-fx[t-2] if t>1 else -1.0
        if t-tb > 10:
            break
        dx[t] = grad(x[t-1])
        
        # Update set size
        det = np.multiply(dx[t], dx[t-1])
        # Increase where det>0
        step[det>0] = step[det>0]*inc_fact
        # Decrease where det<0
        step[det<0] = step[det<0]*dec_fact
        # Upper/lower bound by min/max
        step        = step.clip(step_min, step_max)
        
        # Update w
        # If det >= 0, same as RProp
        x[t][det>=0] = x[t-1][det>=0] - np.multiply(np.sign(dx[t]), step)[det>=0]
        # If func(x[t-1])>func(x[t-2]) set x[t] to x[t-2] where det<0 (only happens if t>1, as det==0 for t=1)
        if delta>0:
            x[t][det<0] = x[t-2][det<0]
        else:
            x[t][det<0] = x[t-1][det<0] - np.multiply(np.sign(dx[t]), step)[det<0]
        # Reset dx[t] = 0 where det<0
        dx[t][det<0] = 0
        
        # Compute func value
        fx[t] = func(x[t])
        if fx[t] < fx[tb]:
            tb = t

        t += 1
    return x[tb]
