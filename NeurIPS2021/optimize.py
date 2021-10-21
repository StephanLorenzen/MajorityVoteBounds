#
# Runs optimization experiments.
#
# Usage: python optimize.py <data_set> [M] [base_clf] [sample_mode] [optimizer] [repeats]
#        M           : number of trees
#        base_clf    : 'rfc', 'abc', 'mce'
#                      'rfc'        = random forest, with fully-grown decision trees
#                      'abc'        = adaboost, with decision stumps
#                      'mce'        = multiple heterogeneous classifier, defined in mvb/mce.py
#        sample_mode : 'bootstrap', 'dim', f in [0,1], 'boost-(int)'
#                      'bootstrap'  = full bagging.
#                      'dim'        = sample d points with replacement
#                      float f      = sample f*|S| points with replacement
#                      'boost-(int)'= boosting with data divided into (int) splits 
#        optimizer   : CMA, GD, RProp, iRProp (default)
#        repeats     : integer
#
# Return: results saved in out
#
import sys
import os
import numpy as np
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import OurAdaBoostClassifier as OABC
from mvb import BaseAdaBoostClassifier as baseABC
from mvb import MultiClassifierEnsemble as MCS
from mvb import data as mldata

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
BASE    = sys.argv[3] if len(sys.argv)>=4 else 'rfc'
SMODE   = sys.argv[4] if len(sys.argv)>=5 else 'bootstrap'
SMODE   = SMODE if (SMODE=='bootstrap' or SMODE=='dim' or 'boost' in SMODE) else float(SMODE)
(SMODE,SPLITS) = ('boost', int(SMODE[-1])) if 'boost' in SMODE else (SMODE, None)
OPT     = sys.argv[5] if len(sys.argv)>=6 else 'iRProp'
REPS    = int(sys.argv[6]) if len(sys.argv)>=7 else 1
if BASE not in ('rfc', 'abc', 'mce'):
    print("No such base classifier")
    sys.exit(1)
    
#if (BASE == 'rfc' and DATASET == 'Protein'):
#    sys.exit(1)

inpath  = 'data/'
outpath = 'out/optimize/'+BASE+'/'

SEED = 1000

if SMODE == 'boost':
    sys.exit(1)

def _write_dist_file(name, rhos, risks):
    with open(outpath+name+'.csv', 'w') as f:
        f.write("h;risk;rho_lam;rho_tnd;rho_mu;rho_bern\n")
        for i,(err,r_lam,r_tnd,r_mu,r_bern) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2], rhos[3])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_tnd)+";"+str(r_mu)+";"+str(r_bern)+"\n")        

if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

def _write_outfile(results):
    prec = 5
    with open(outpath+DATASET+'-'+str(M)+'-'+str(SMODE)+'-'+str(OPT)+'.csv', 'w') as f:
        # Header
        f.write('repeat;n_train;n_test;d;c')
        for name in ["best","unf","lam","tnd","mu","bern"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem', 'sh', 'pbkl', 'tnd', 'TandemUB', 'MU', 'ub_tr', 'lb_gr', 'muTandemUB', 'bern', 'mutandem_risk', 'vartandem_risk', 'KL', 'varUB', 'bernTandemUB', 'bl', 'bg', 'bmu']]))
        f.write('\n')
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3]));
            for (mv_risk, stats, bounds, bl, bg, bm) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(20)]))
                        .format(mv_risk,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('SH', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('TND', -1.0),
                            stats.get('TandemUB', -1.0), # TandemUB is the empirical bound of tandem risk; 4*TandemBound = TND
                            bounds.get('MU', -1.0),
                            stats.get('ub_tr', -1.0),   # kl^{-1}+ of tandem risk in the C\muTND bound
                            stats.get('lb_gr', -1.0),   # kl^{-1}- of gibbs risk in the C\muTND bound
                            stats.get('muTandemUB', -1.0), # muTandemUB is the empirical bound for the numerator in Eq.(1)
                            bounds.get('MUBernstein', -1.0),
                            stats.get('mutandem_risk', -1.0),
                            stats.get('vartandem_risk', -1.0),
                            stats.get('KL', -1.0),
                            stats.get('varUB', -1.0), # varUB is the empirical bound of the variance (Corollary 17)
                            stats.get('bernTandemUB', -1.0), # bernTandemUB is the empirical bound of the mu tandem risk (Corollary 20); bernTandemUB/(1/2-mu)**2 = MUBernstein
                            bl,
                            bg,
                            bm
                                )
                        )
            f.write('\n')
        

if SMODE=='bootstrap':
    smodename = 'bagging'
elif SMODE=='boost':
    smodename = 'boosting with ' + str(SPLITS) + ' splits'
else:
    smodename = 'reduced bagging ('+str(SMODE)+');'
print("Starting "+BASE.upper()+ " optimization (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename+", optimizer = "+str(OPT))
results = []
X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]
for rep in range(REPS):
    if REPS>1:
        print("####### Repeat = "+str(rep+1))

    rhos = []
    # define the range of mu  for 'C$\mu$TND' and 'COTND'
    mu_range = (-0.5, 0.5)
    
    # Prepare Data
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

    # Prepare base classifiers for PAC-Bayes methods
    if BASE == 'rfc':
        rf = RFC(M,max_features="sqrt",random_state=RAND, sample_mode=SMODE)
    elif BASE == 'mce':
        rf = MCS(n_estimators=M, random_state=RAND, sample_mode=SMODE)
    else:
        max_depth = 1
        rf = OABC(n_estimators=M, max_depth=max_depth, random_state=RAND, sample_mode=SMODE, n_splits = SPLITS, use_ada_prior=True)
        abc = baseABC(n_estimators=M, max_depth=max_depth, random_state=RAND, n_splits = SPLITS)
        
    # Training
    print("Training...")    
    _ = rf.fit(trainX,trainY)
    
    """ Especially for MCE """
    print('Best model in the ensemble...')
    _ = rf.optimize_rho('Best')
    _, mv_risk = rf.predict(testX,testY)
    stats  = rf.stats()
    stats = rf.aggregate_stats(stats) # initial stats after training
    bounds, stats = rf.bounds(stats=stats)
    res_best = (mv_risk, stats, bounds, -1, -1, -1)
    print('mv_risk', mv_risk)
    
    """ Uniform Weighting """
    print('Uniform weighting...')
    _ = rf.optimize_rho('Uniform')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_unf = (mv_risk, stats, bounds, -1, -1, -1)
    print('mv_risk', mv_risk)
      
    """ Optimize Lambda """
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_bern = (0.0,)
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_lam = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk)

        
    """ Optimize TND """
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_bern = (0.0,)
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_tnd = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk)

    """ Optimize C$\mu$TND with grid """
    print("Optimizing C\mu TND...")
    (_, rho, bmu, bl, bg) = rf.optimize_rho('MU', options={'optimizer':OPT})
    print('bmu', bmu)
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_kl':(bmu,)}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mu
    res_MU = (mv_risk, stats, bounds, bl, bg, bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk)
    
    
    """ Optimize COTND with grid by Binary Search """
    print("Optimizing COTND (using binary search) in ({}, {})".format(str(mu_range[0]), str(mu_range[1])))
    (_, rho, bmu, bl, bg) = rf.optimize_rho('MUBernstein', options={'optimizer':OPT,'mu_bern':mu_range})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_bern':(bmu,), 'lam':bl, 'gam': bg}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_Bern = (mv_risk, stats, bounds, bl, bg, bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk)
     
    # opt = (bound, rho, lam, gam, mu)
    if rep==0:
        # record the \rho distribution by all optimization methods
        _write_dist_file('rho-'+DATASET, rhos, stats['risks'])

    results.append((rep, n, (res_best, res_unf, res_lam, res_tnd, res_MU, res_Bern)))

_write_outfile(results)

