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
from mvb import data as mldata

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
SMODE   = sys.argv[3] if len(sys.argv)>=4 else 'bootstrap'
SMODE   = SMODE if (SMODE=='bootstrap' or SMODE=='dim') else float(SMODE)
OPT     = sys.argv[4] if len(sys.argv)>=5 else 'iRProp'
REPS    = int(sys.argv[5]) if len(sys.argv)>=6 else 1

inpath  = 'data/'
outpath = 'out/optimize/'

SEED = 1000

def _write_dist_file(name, rhos, risks):
    with open(outpath+name+'.csv', 'w') as f:
        f.write("h;risk;rho_lam;rho_tnd;rho_mu;rho_mug\n")
        for i,(err,r_lam,r_tnd,r_mu,r_mug) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2], rhos[3])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_tnd)+";"+str(r_mu)+";"+str(r_mug)+"\n")

if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

def _write_outfile(results):
    prec = 5
    with open(outpath+DATASET+'-'+str(M)+'-'+str(SMODE)+'-'+str(OPT)+'.csv', 'w') as f:
        f.write('repeat;n_train;n_test;d;c')
        for name in ["unf","lam","tnd","mu","mug", "muBernsteing"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk','gibbs','tandem_risk','pbkl','c1','c2','ctd','tnd','mub','lamda','gamma','mu']]))
        f.write('\n')
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3]));
            for (mv_risk, stats, bounds, bl, bg, bm) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(13)]))
                        .format(mv_risk,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('C1', -1.0),
                            bounds.get('C2', -1.0),
                            bounds.get('CTD', -1.0),
                            bounds.get('TND', -1.0),
                            bounds.get('MU',-1.0),
                            bounds.get('MUBernstein', -1.0),
                            bl,
                            bg,
                            bm
                            )
                        )
            f.write('\n')


smodename = 'bagging' if SMODE=='bootstrap' else ('reduced bagging ('+str(SMODE)+');')
print("Starting RFC optimization (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename+", optimizer = "+str(OPT))
results = []
X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]
for rep in range(REPS):
    if REPS>1:
        print("####### Repeat = "+str(rep+1))
    
    rf = RFC(M,max_features="sqrt",random_state=RAND, sample_mode=SMODE)
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)
    
    rhos = []
    print("Training...")
    _  = rf.fit(trainX,trainY)
    _, mv_risk = rf.predict(testX,testY)
    stats  = rf.stats()
    
    bounds = rf.bounds(stats=stats)
    res_unf = (mv_risk, stats, bounds, -1, -1, -1)
        
    # Optimize Lambda
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_lam = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
        
    # Optimize TND
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_mv2 = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    
    # Optimize MU
    print("Optimizing MU...")
    (_, rho, bl, bg, mu) = rf.optimize_rho('MU', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_mu = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)
    
    # Optimize MU with grid
    print("Optimizing MU (using grid)...")
    grid = [-0.2,-0.1,0.0,0.1]
    (_, rho, mu, bl, bg) = rf.optimize_rho('MU', options={'optimizer':OPT,'mu_grid':grid})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_mug = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)
    
    # Optimize MUBernstein with grid
    print("Optimizing MUBernstein (using grid)...")
    grid = [-0.2,-0.1,0.0,0.1]
    (_, rho, mu, bl, bg) = rf.optimize_rho('MUBernstein', options={'optimizer':OPT,'mu_grid':grid})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_muBernsteing = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)


    # opt = (bound, rho, lam, gam, mu)
    if rep==0:
        _write_dist_file('rho-'+DATASET, rhos, stats['risks'])
    results.append((rep, n, (res_unf, res_lam, res_mv2, res_mu, res_mug, res_muBernsteing)))
    
_write_outfile(results)

