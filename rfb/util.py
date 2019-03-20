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

def compute_kl(rho,pi):
    m = pi.shape[0]
    assert(rho.shape[0]==m)
    kl = 0.0
    for h in range(m):
        kl += rho[h]*log(rho[h]/pi[h])
    return kl

def uniform_distribution(m):
    return (1.0/m) * np.ones((m,))



######################
# Risk computation

# Simply computes error
# preds.shape = (n,)
# targs.shape = (n,)
def compute_risk(preds, targs):
    assert(preds.shape == targs.shape)
    return 1.0-accuracy_score(preds,targs)


# Accepts:
#    -> a set of m OOB-samples with predictions, and targets,
# and/or
#    -> a validation set with preds for m trees, and targets.
# and a distribution rho.
#
# oob_set = ([(idx,preds)]*m, targs)
#  where
#    idx.shape = preds.shape (elements (idx) and preds for OOB sample for each tree)
#    targs.shape = (n,)
#
# val_set = (preds, targs)
#  where
#    preds.shape = (n,m)
#    targs.shape = (n,)
#
#
# Returns dict containing following empiricals:
#    risk_gibbs:   Gibbs risk
#    risk_mv:      MV risk on val_set (if val_set != None)
#    error_list:   Risk for each tree
#    n_min:        Min size for set used to evaluate any tree
#    disagreement: Disagreement
#    dis_mat:      Disagreement matrix
#    joint_error:  Joint error
#    jerr_mat:     Joint error matrix
#    jn_min:       Min size for set used to evaluate any pair of trees
def compute_stats(oob_set=None, val_set=None, rho=None):
    # Validation
    assert(not (oob_set is None and val_set is None))
    
    m,n_val = 0,0
    doOOB, doVal = oob_set is not None, val_set is not None
    if doOOB:
        assert(len(oob_set)==2)
        preds, targs = oob_set
        m = len(preds)
    
    if doVal:
        assert(len(val_set)==2)
        preds, targs = val_set
        assert(m==0 or m==preds.shape[0])
        m     = preds.shape[0]
        n_val = targs.shape[0]
        assert(preds.shape[1]==n_val)

    if rho is None:
        rho = uniform_distribution(m)
    assert(rho.shape[0] == m)
    
    error_list = np.zeros((m,))
    count_list = np.zeros((m,))
    dis_mat    = np.zeros((m,m))
    jerr_mat   = np.zeros((m,m))
    count_mat  = np.zeros((m,m))

    if doOOB:
        preds, targs = oob_set
        for i in range(m):
            (i_idx, i_preds) = preds[i]
            i_targs = targs[i_idx]
            cnt = i_preds.shape[0]
            error_list[i] += np.sum(i_preds!=i_targs)
            count_list[i] += cnt

            for j in range(i, m):
                (j_idx, j_preds) = preds[j]
                i_mp = dict(zip(i_idx,i_preds))
                j_mp = dict(zip(j_idx,j_preds))

                dis, jerr, cnt = 0.0, 0.0, 0.0
                for idx in set(i_mp.keys()) & set(j_mp.keys()): # set(...) wrapper for P2.7 compat
                    y   = targs[idx]
                    i_p = i_mp[idx]
                    j_p = j_mp[idx]
                    dis  += 1.0 if i_p != j_p else 0.0
                    jerr += 1.0 if i_p != y and j_p != y else 0.0
                    cnt  += 1.0

                dis_mat[i,j]  += dis
                dis_mat[j,i]  += dis
                jerr_mat[i,j] += jerr
                jerr_mat[j,i] += jerr
                count_mat[i,j] += cnt
                count_mat[j,i] += cnt
 
    if doVal:
        preds, targs = val_set
        for i in range(m):
            i_preds = preds[i]
            
            error_list[i] += np.sum(i_preds!=targs)
            count_list[i] += n_val
            
            for j in range(i, m):
                j_preds = preds[j]
                
                dis  = np.sum(i_preds != j_preds)
                jerr = np.sum(np.logical_and((i_preds!=targs), (j_preds!=targs)))

                dis_mat[i,j]  += dis
                dis_mat[j,i]  += dis
                jerr_mat[i,j] += jerr
                jerr_mat[j,i] += jerr
                count_mat[i,j] += n_val
                count_mat[j,i] += n_val


    error_list /= count_list
    dis_mat /= count_mat
    jerr_mat /= count_mat

#    risk_gibbs:   Gibbs risk
#    risk_mv:      MV risk on val_set (if val_set != None)
#    error_list:   Risk for each tree
#    n_min:        Min size for set used to evaluate any tree
#    disagreement: Disagreement
#    dis_mat:      Disagreement matrix
#    joint_error:  Joint error
#    jerr_mat:     Joint error matrix
#    jn_min:       Min size for set used to evaluate any pair of trees

    result = dict()
    result['risk_gibbs']   = np.average(error_list, weights=rho)
    result['error_list']   = error_list
    result['n_min']        = np.min(count_list)
    result['disagreement'] = np.average(np.average(dis_mat, weights=rho, axis=0), weights=rho)
    result['dis_mat']      = dis_mat
    result['joint_error']  = np.average(np.average(jerr_mat, weights=rho, axis=0), weights=rho)
    result['jerr_mat']     = jerr_mat
    result['jn_min']       = np.min(count_mat)

    if doVal:
        preds, targs = val_set
        result['risk_mv'] = compute_mv_risk(rho, preds, targs)

    return result

#######################
# OOB estimate

def compute_oob_estimate(rho, oob_set):
    assert(len(oob_set)==2)
    preds, targs = oob_set
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
    for i, pl in enumerate(Ps):
        assert(len(pl)>0)
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
    return compute_risk(results, targs)

#######################
# Majority vote risk

def compute_mv_risk(rho, preds, targs):
    mvPreds = compute_mv_preds(rho, preds)
    return compute_risk(mvPreds, targs) 

def compute_mv_preds(rho, preds):
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


