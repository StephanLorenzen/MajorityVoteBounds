#
# Runs optimization experiments.
#
# Usage: python optimize.py <data_set> [M] [sample_mode] [optimizer] [repeats]
#        M           : number of trees
#        sample_mode : 'bootstrap', 'dim', f in [0,1]
#                      'bootstrap' = full bagging.
#                      'dim'       = sample d points with replacement
#                      float f     = sample f*|S| points with replacement
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
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.ensemble import AdaBoostClassifier
from mvb import data as mldata

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
SMODE   = sys.argv[3] if len(sys.argv)>=4 else 'bootstrap'
SMODE   = SMODE if (SMODE=='bootstrap' or SMODE=='dim' or SMODE=='boost') else float(SMODE)
OPT     = sys.argv[4] if len(sys.argv)>=5 else 'iRProp'
REPS    = int(sys.argv[5]) if len(sys.argv)>=6 else 1

inpath  = 'data/'
outpath = 'out/optimize/'

SEED = 1000

DIFF_DATASET = [
        'Protein',
        'Connect-4',
        'Pendigits',
        'Letter',
        'SatImage',
        'Sensorless',
        'USPS',
        'MNIST',
        'Fashion-MNIST',
]
#max_depth = 2 if DATASET in DIFF_DATASET else 1
max_depth = 1

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
        for name in ["ada","unf","lam","tnd","mu","bern"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem', 'sh', 'pbkl', 'tnd', 'TandemUB', 'MU', 'muTandemUB', 'bern', 'mutandem_risk', 'vartandem_risk', 'KL', 'varUB', 'bernTandemUB', 'bl', 'bg', 'bmu']]))
        f.write('\n')
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3]));
            for (mv_risk, stats, bounds, bl, bg, bm) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(18)]))
                        .format(mv_risk,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('tandem_risk', -1.0),
                            #stats.get('disagreement', -1.0),
                            bounds.get('SH', -1.0),
                            bounds.get('PBkl', -1.0),
                            #bounds.get('C1', -1.0),
                            #bounds.get('C2', -1.0),
                            #bounds.get('CTD', -1.0),
                            bounds.get('TND', -1.0),
                            stats.get('TandemUB', -1.0), # TandemUB is the empirical bound of tandem risk; 4*TandemBound = TND
                            bounds.get('MU', -1.0),
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
        


smodename = 'bagging' if (SMODE=='bootstrap' or SMODE=='boost') else ('reduced bagging ('+str(SMODE)+');')
print("Starting RFC optimization (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename+", optimizer = "+str(OPT))
results = []
X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]
for rep in range(REPS):
    if REPS>1:
        print("####### Repeat = "+str(rep+1))

    rhos = []
    # define the range of mu 'MUBernstein'
    mu_range = (-0.5, 0.5)
    
    # Prepare Data
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

    """
    # Adaboost Baseline
    print("Calculate the baseline by AdaBoost...")
    base_estimator = Tree(max_depth=max_depth)
    abc = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=M, algorithm='SAMME', random_state=RAND)
    abc.fit(trainX, trainY)
    abc_risk = 1.0 - abc.score(testX, testY)
    print('Baseline:', abc_risk)
    """
    # Prepare base classifiers for PAC-Bayes methods
    #rf = RFC(M,max_features="sqrt",random_state=RAND, sample_mode=SMODE)
    rf = OABC(n_estimators=M, max_depth=max_depth, random_state=RAND, sample_mode=SMODE)
    
    # Training
    print("Training...")
    #_  = rf.fit(trainX,trainY) # for non-boosting
    (valX,valY) = rf.fit(trainX,trainY) # for boosting
    
    # Adaboost with bound
    print("Adaboost with bound...")
    rho = rf.optimize_rho('AdaBoost')
    _, mv_risk = rf.predict(testX, testY)
    stats = rf.stats(labeled_data=(valX,valY))
    bounds, _ = rf.bounds(stats=stats)
    res_ada = (mv_risk, stats, bounds, -1, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'bound', bounds['SH'])
    
    # Uniform Weighting
    print('Uniform weighting...')
    _ = rf.optimize_rho('Uniform')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats,options = {'mu_kl':mu_range,'mu_bern': mu_range}) # initial stats after training
    bounds, stats = rf.bounds(stats=stats) # compute the bounds according to the best mu in the range, and record the corresponding stats
    res_unf = (mv_risk, stats, bounds, -1, -1, -1)
    print('mv_risk', mv_risk, 'KL', res_unf[1]['KL'])
      
    # Optimize Lambda
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_bern = (0.0,)
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_lam = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'KL', res_lam[1]['KL'])

        
    # Optimize TND
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_bern = (0.0,)
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_tnd = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'KL', res_tnd[1]['KL'])

    # Optimize MU with grid by Binary Search
    print("Optimizing MU (using binary search) in ({}, {})".format(str(mu_range[0]), str(mu_range[1])))
    (_, rho, bmu, bl, bg) = rf.optimize_rho('MU', options={'optimizer':OPT,'mu_kl':mu_range})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_kl':(bmu,)}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_MU = (mv_risk, stats, bounds, bl, bg, bmu)
    print('Bern bound: gamma, ', bg, 'lambda', bl, 'mu', bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'KL', res_MU[1]['KL'])
    
    
    # Optimize MUBennett with grid by Binary Search
    print("Optimizing MUBennett (using binary search) in ({}, {})".format(str(mu_range[0]), str(mu_range[1])))
    (_, rho, bmu, bl, bg) = rf.optimize_rho('MUBernstein', options={'optimizer':OPT,'mu_bern':mu_range})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_bern':(bmu,)}) # update rho-dependent stats
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_Bern = (mv_risk, stats, bounds, bl, bg, bmu)
    print('Bern bound: gamma, ', bg, 'lambda', bl, 'mu', bmu)
    rhos.append(rho)
    print('mv_risk', mv_risk, 'KL', res_Bern[1]['KL'])
     


    # opt = (bound, rho, lam, gam, mu)
    if rep==0:
        # record the \rho distribution by all optimization methods
        _write_dist_file('rho-'+DATASET, rhos, stats['risks'])
    results.append((rep, n, (res_ada, res_unf, res_lam, res_tnd, res_MU, res_Bern)))

_write_outfile(results)

