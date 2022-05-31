#
# Runs optimization experiments.
#
# Usage: python optimize.py <data_set> [M] [base_clf] [sample_mode] [optimizer] [repeats]
#        M           : number of trees
#        base_clf    : 'rfc', 'abc', 'mce'
#                      'rfc'        = random forest, with fully-grown decision trees
#                      'abc'        = adaboost, with decision stumps (NOT MAINTAINED)
#                      'mce'        = multiple heterogeneous classifier, defined in mvb/mce.py
#        sample_mode : 'bootstrap', 'dim', f in [0,1], 'boost-(int)'
#                      'bootstrap'  = full bagging.
#                      'dim'        = sample d points with replacement
#                      float f      = sample f*|S| points with replacement
#                      'boost-(int)'= boosting with data divided into (int) splits (NOT MAINTAINED)
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

if BASE not in ('rfc', 'mce'):
    print("No such base classifier")
    sys.exit(1)

if SMODE == 'boost':
    sys.exit(1)
    
if (BASE == 'rfc' and DATASET == 'Protein'):
    sys.exit(1)


inpath  = 'data/'
outpath = 'out/optimize/'+BASE+'/'
if not os.path.exists(outpath):
    os.makedirs(outpath)


SEED = 1000
RAND = check_random_state(SEED)

""" Write the files to plot rhos.
    PLOT: plots/figure/"+BASE+"/plot_rhos.pdf
"""
def _write_dist_file(name, rhos, risks):
    with open(outpath+name+'.csv', 'w') as f:
        f.write("h;risk;rho_lam;rho_tnd;rho_cctnd;rho_ccpbb\n")
        for i,(err,r_lam,r_tnd,r_cctnd,r_ccpbb) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2], rhos[3])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_tnd)+";"+str(r_cctnd)+";"+str(r_ccpbb)+"\n")        

""" Write the files for figures and tables of the experiments
    PLOT:  plots/figure/"+BASE+"/mvrisk_optimized_comparison.pdf
           plots/figure/"+BASE+"/bounds_optimized_comparison.pdf
    TABLE: plots/table/"+BASE+"/optimize_test_risk.pdf
           plots/table/"+BASE+"/optimize_bound.pdf
           plots/table/"+BASE+"/optimize_test_values.pdf
"""
def _write_outfile(results):
    prec = 5
    with open(outpath+DATASET+'-'+str(M)+'-'+str(SMODE)+'-'+str(OPT)+'.csv', 'w') as f:
        # Header
        f.write('repeat;n_train;n_test;d;c')
        for name in ["best","unf","lam","tnd","cctnd","ccpbb"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem', 'SH', 'PBkl', 'TND', 'CCTND', 'ub_tr', 'eb_gr', 'CCPBB', 'mutandem', 'varmutandem', 'kl', 'ub_var', 'ub_mutandem', 'bl', 'bg', 'bmu']]))
        f.write('\n')
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3]));
            for (mv_risk, stats, bounds, bl, bg, bm) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(18)]))
                        .format(mv_risk,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('SH', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('TND', -1.0),
                            bounds.get('CCTND', -1.0),
                            stats.get('ub_tr', -1.0),   # kl^{-1}+ of tandem risk in the CCTND bound
                            stats.get('eb_gr', -1.0),   # kl^{-1} of gibbs risk in the CCTND bound
                            bounds.get('CCPBB', -1.0),
                            stats.get('mutandem_risk', -1.0),
                            stats.get('vartandem_risk', -1.0),
                            stats.get('KL', -1.0),
                            stats.get('ub_var', -1.0), # empirical bound of the variance (Thm 13)
                            stats.get('ub_mutandem', -1.0), # numerator of Thm 15
                            bl,
                            bg,
                            bm
                                )
                        )
            f.write('\n')

""" Start the experiment. """

if SMODE=='bootstrap':
    smodename = 'bagging'
elif SMODE=='boost':
    smodename = 'boosting with ' + str(SPLITS) + ' splits'
else:
    smodename = 'reduced bagging ('+str(SMODE)+');'
print("Starting "+BASE.upper()+ " optimization (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename+", optimizer = "+str(OPT))

results = [] # Record the results for _write_outfile()
X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]
for rep in range(REPS):
    if REPS>1:
        print("####### Repeat = "+str(rep+1))

    rhos = [] # Record the rhos for _write_dist_file()
    
    """ Prepare Data """
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

    """ Prepare base classifiers for PAC-Bayes methods """
    if BASE == 'rfc':
        rf = RFC(M,max_features="sqrt",random_state=RAND, sample_mode=SMODE)
    elif BASE == 'mce':
        rf = MCS(n_estimators=M, random_state=RAND, sample_mode=SMODE)
    else:
        sys.exit(1)
        #max_depth = 1
        #rf = OABC(n_estimators=M, max_depth=max_depth, random_state=RAND, sample_mode=SMODE, n_splits = SPLITS, use_ada_prior=True)
        #abc = baseABC(n_estimators=M, max_depth=max_depth, random_state=RAND, n_splits = SPLITS)
        
    """ Training """
    print("Training...")    
    _ = rf.fit(trainX,trainY)
    
    """ Especially for MCE """
    print('Best model in the ensemble...')
    _ = rf.optimize_rho('Best')
    _, mv_risk = rf.predict(testX,testY)
    stats  = rf.stats() # initial stats after training
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_best = (mv_risk, stats, bounds, -1, -1, -1)
    print('mv_risk', mv_risk)
    
    """ Uniform Weighting """
    print('Uniform weighting...')
    _ = rf.optimize_rho('Uniform')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats)
    res_unf = (mv_risk, stats, bounds, -1, -1, -1)
    print('mv_risk', mv_risk)
      
    """ Optimize Lambda """
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats, spec_bound='PBkl')
    res_lam = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk)

    """ Optimize TND """
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats, spec_bound='TND')
    res_tnd = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'bound', bounds.get('TND', -1.0))

    """ Optimize CCTND """
    print("Optimizing CCTND...")
    (_, rho, bmu, bl, bg) = rf.optimize_rho('CCTND', options={'optimizer':OPT})
    print('bmu ', bmu)
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_CCTND':bmu}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats, spec_bound='CCTND') # compute the CCTND bound with bmu
    res_cctnd = (mv_risk, stats, bounds, bl, bg, bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'bound', bounds.get('CCTND', -1.0))
    
    """ Optimize CCPBB with grid by Binary Search """
    # define the range of mu  for 'CCPBB'
    mu_range = (-0.5, 0.5)
    print("Optimizing CCPBB (using binary search) in ({}, {})".format(str(mu_range[0]), str(mu_range[1])))
    (_, rho, bmu, bl, bg) = rf.optimize_rho('CCPBB', options={'optimizer':OPT,'mu_CCPBB':mu_range})
    print('bmu ', bmu)
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_CCPBB':bmu, 'lam':bl, 'gam': bg}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats, spec_bound='CCPBB') # compute the bound with the above parameters
    res_ccpbb = (mv_risk, stats, bounds, bl, bg, bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'bound', bounds.get('CCPBB', -1.0))
    
    if rep==0:
        # record the \rho distribution by all optimization methods
        _write_dist_file('rho-'+DATASET, rhos, stats['risks'])

    results.append((rep, n, (res_best, res_unf, res_lam, res_tnd, res_cctnd, res_ccpbb)))

_write_outfile(results)

