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
    for j, (idx, P) in enumerate(preds):
        risks[j] = np.sum(P!=targs[idx])
        ns[j] = idx.shape[0]
    return risks, ns

def oob_disagreements(preds):
    m = len(preds)

    disagreements = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (i_idx, i_preds) = preds[i]
        for j in range(i, m):
            (j_idx, j_preds) = preds[j]
            i_mp = dict(zip(i_idx,i_preds))
            j_mp = dict(zip(j_idx,j_preds))

            dis, c = 0, 0
            for idx in set(i_mp.keys()) & set(j_mp.keys()): # set(...) wrapper for P2.7 compat
                i_p = i_mp[idx]
                j_p = j_mp[idx]
                dis  += 1 if i_p != j_p else 0
                c    += 1

            disagreements[i,j] += dis
            n2[i,j]            += c
            if i != j:
                disagreements[j,i]  += dis
                n2[j,i]             += c
    
    return disagreements, n2    

def oob_tandem_risks(preds, targs):
    m = len(preds)

    tandem_risks  = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (i_idx, i_preds) = preds[i]
        for j in range(i, m):
            (j_idx, j_preds) = preds[j]
            i_mp = dict(zip(i_idx,i_preds))
            j_mp = dict(zip(j_idx,j_preds))

            tand, c = 0, 0
            for idx in set(i_mp.keys()) & set(j_mp.keys()): # set(...) wrapper for P2.7 compat
                y   = targs[idx]
                i_p = i_mp[idx]
                j_p = j_mp[idx]
                tand += 1 if i_p != y and j_p != y else 0
                c    += 1

            tandem_risks[i,j]  += tand
            n2[i,j]            += c
            if i != j:
                tandem_risks[j,i]   += tand
                n2[j,i]             += c
    
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
