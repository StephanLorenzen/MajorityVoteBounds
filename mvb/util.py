import numpy as np
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

######################
# Various stats

# Simply computes error
# preds.shape = (n,)
# targs.shape = (n,)
def risk(preds, targs):
    assert(preds.shape == targs.shape)
    return 1.0-accuracy_score(preds,targs)

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

    results = np.zeros((n,))
    for i,pl in enumerate(preds):
        bins = dict()
        for r,vote in zip(rho,pl):
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
    return results
