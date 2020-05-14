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

def gibbs(preds, targs):
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
def stats(oob_set=None, val_set=None, rho=None, unlabelled=None):
    # Validation
    assert(not (oob_set is None and val_set is None))
    
    m,n_val,n_ext = 0,0,0
    doOOB, doVal, doUlb = oob_set is not None, val_set is not None, unlabelled is not None
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
    
    if doUlb:
        n_ext = unlabelled.shape[0]

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
                jerr_mat[i,j] += jerr
                count_mat[i,j] += cnt
                if i != j:
                    dis_mat[j,i]  += dis
                    jerr_mat[j,i] += jerr
                    count_mat[j,i] += cnt
    
    var_error = 0.0
    if doVal:
        preds, targs = val_set
        
        l_data_errs = np.zeros((targs.shape[0],))
        
        for i in range(m):
            i_preds = preds[i]
            
            error_list[i] += np.sum(i_preds!=targs)
            count_list[i] += n_val
       
            l_data_errs += rho[i]*(i_preds!=targs)
            
            for j in range(i, m):
                j_preds = preds[j]
                
                dis  = np.sum(i_preds != j_preds)
                jerr = np.sum(np.logical_and((i_preds!=targs), (j_preds!=targs)))

                dis_mat[i,j]  += dis
                jerr_mat[i,j] += jerr
                count_mat[i,j] += n_val
                if i != j:
                    dis_mat[j,i]  += dis
                    jerr_mat[j,i] += jerr
                    count_mat[j,i] += n_val
        
        var_error = np.var(l_data_errs)
    

    udis_mat = np.copy(dis_mat)
    error_list /= count_list
    jerr_mat /= count_mat
    dis_mat /= count_mat

    jn_min = np.min(count_mat)

    if doUlb:
        for i in range(m):
            i_preds = unlabelled[i]
            for j in range(i,m):
                j_preds = unlabelled[j]

                dis = np.sum(i_preds != j_preds)

                udis_mat[i,j] += dis
                count_mat[i,j] += n_ext
                if i != j:
                    udis_mat[j,i] += dis
                    count_mat[j,i] += n_ext
                
    udis_mat /= count_mat

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
    result['risk_gibbs']    = np.average(error_list, weights=rho)
    result['error_list']    = error_list
    result['n_min']         = np.min(count_list)
    result['disagreement']  = np.average(np.average(dis_mat, weights=rho, axis=0), weights=rho)
    result['udisagreement'] = np.average(np.average(udis_mat, weights=rho, axis=0), weights=rho)
    result['dis_mat']       = dis_mat
    result['udis_mat']      = udis_mat
    result['joint_error']   = np.average(np.average(jerr_mat, weights=rho, axis=0), weights=rho)
    result['jerr_mat']      = jerr_mat
    result['jn_min']        = jn_min
    result['un_min']        = jn_min+n_ext # #joint_min + #unlabelled
    result['var_error']     = var_error
    result['n_val']         = n_val
    result['n_ext']         = n_ext

    if doVal:
        preds, targs = val_set
        result['risk_mv'] = mv_risk(rho, preds, targs)

    return result

#######################
# OOB 

# Accepts:
#    -> a set of m OOB-samples with predictions, and targets,
#
# oob_set = ([(idx,preds)]*m, targs)
#  where
#    idx.shape = preds.shape (elements (idx) and preds for OOB sample for each tree)
#    targs.shape = (n,)
#
# Returns:
#    -> Five tuple: risks, n, disagreements, tandem_risks, n2
def oob_stats(oob_preds, Y):
    m = len(oob_preds)

    risks = np.zeros((m,))
    n = np.zeros((m,))
    disagreements = np.zeros((m,m))
    tandem_risks  = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (i_idx, i_preds) = oob_preds[i]
        i_targs = Y[i_idx]
        
        risks[i] += np.sum(i_preds!=i_targs)
        n[i]     += i_preds.shape[0]

        for j in range(i, m):
            (j_idx, j_preds) = oob_preds[j]
            i_mp = dict(zip(i_idx,i_preds))
            j_mp = dict(zip(j_idx,j_preds))

            dis, tand, c = 0, 0, 0
            for idx in set(i_mp.keys()) & set(j_mp.keys()): # set(...) wrapper for P2.7 compat
                y   = Y[idx]
                i_p = i_mp[idx]
                j_p = j_mp[idx]
                dis  += 1 if i_p != j_p else 0
                tand += 1 if i_p != y and j_p != y else 0
                c    += 1

            disagreements[i,j] += dis
            tandem_risks[i,j]  += tand
            n2[i,j]            += c
            if i != j:
                disagreements[j,i]  += dis
                tandem_risks[j,i]   += tand
                n2[j,i]             += c
    
    return risks, n, disagreements, tandem_risks, n2    

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
