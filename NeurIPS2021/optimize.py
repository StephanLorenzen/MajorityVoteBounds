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
        # Header
        f.write('repeat;n_train;n_test;d;c')
        for name in ["unf","lam","tnd","mug","MUBernsteing"]:
            if name == "unf":
                # TandemUB is the empirical bound of tandem risk; 4*TandemBound = tnd
                # muTandemUB is the empirical bound of mu tandem risk by Stephan; muTandemBound/(1/2-mu)**2 = mub
                # varUB is the empirical bound of the variance (Corollary 17)
                # bernTandemUB is the empirical bound of the mu tandem risk (Corollary 20); bernTandemUB/(1/2-mu)**2 = bern
                # 'mu_mub' is the optimal \mu when using the 1st mu bound; 
                # 'mu_bern' is the optimal \mu when using the bernstein bound; 
                # 'mu_est' is the estimation of \mu using some samples in oobs
                f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem_risk', 'pbkl', 'c1', 'c2', 'ctd', 'tnd', 'TandemUB', 'mub', 'mu_mub', 'muTandemUB', 'bern', 'mu_bern', 'mutandem_risk', 'vartandem_risk', 'varUB', 'bernTandemUB']]))
            elif name == "lam":
                f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem_risk', 'pbkl', 'lambda']]))
            elif name == "tnd":
                f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem_risk', 'tnd', 'lambda']]))
            elif name == "mug":
                f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem_risk', 'mub', 'mu_mub', 'muTandemUB', 'lambda', 'gamma']]))
            elif name == "MUBernsteing":
                # 'tandem_risk' here is the mutandem_risk with the optimal \mu
                # 'vartandem_risk' is the vartandem_risk with the optimal \mu
                # 'tandem_bound' is the bound obtained by _muBernstein
                # 'vartandem_bound' is the bound obtained _varMUBernstein
                f.write(';'+';'.join([name+'_'+x for x in ['mv_risk', 'gibbs', 'tandem_risk', 'bern', 'mu_bern', 'mutandem_risk', 'vartandem_risk', 'varUB', 'bernTandemUB', 'lambda', 'gamma']]))
        f.write('\n')
        
        # report the results for each repetition
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3]));
            # restup[0] : res_unf
            (mv_risk, stats, bounds, bl, bg, mu) = restup[0]
            f.write(
                    (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(18)]))
                    .format(mv_risk,
                        stats.get('gibbs_risk', -1.0),
                        stats.get('tandem_risk', -1.0),
                        bounds.get('PBkl', -1.0),
                        bounds.get('C1', -1.0),
                        bounds.get('C2', -1.0),
                        bounds.get('CTD', -1.0),
                        bounds.get('TND', -1.0),
                        stats.get('TandemUB', -1.0),
                        bounds.get('MU',-1.0),
                        stats['mu_mub'][0],
                        stats.get('muTandemUB', -1.0),
                        bounds.get('MUBernstein', -1.0),
                        stats['mu_bern'][0],
                        stats.get('mutandem_risk', -1.0),
                        stats.get('vartandem_risk', -1.0),
                        stats.get('varUB', -1.0),
                        stats.get('bernTandemUB', -1.0)
                            )
                    )
            # restup[1] : res_lam
            (mv_risk, stats, bounds, bl, bg, mu) = restup[1]
            f.write(
                    (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(5)]))
                    .format(mv_risk,
                        stats.get('gibbs_risk', -1.0),
                        stats.get('tandem_risk', -1.0),
                        bounds.get('PBkl', -1.0),
                        bl
                            )
                    )
            # restup[2] : res_tnd
            (mv_risk, stats, bounds, bl, bg, mu) = restup[2]
            f.write(
                    (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(5)]))
                    .format(mv_risk,
                        stats.get('gibbs_risk', -1.0),
                        stats.get('tandem_risk', -1.0),
                        bounds.get('TND', -1.0),
                        bl
                            )
                    )
            # restup[3] : res_mub
            (mv_risk, stats, bounds, bl, bg, mu) = restup[3]
            f.write(
                    (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(8)]))
                    .format(mv_risk,
                        stats.get('gibbs_risk', -1.0),
                        stats.get('tandem_risk', -1.0),
                        bounds.get('MU',-1.0),
                        mu,
                        stats.get('muTandemUB', -1.0),
                        bl,
                        bg
                            )
                    )
            # restup[4] : res_muMUBersteing
            (mv_risk, stats, bounds, bl, bg, mu) = restup[4]
            f.write(
                    (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(11)]))
                    .format(mv_risk,
                        stats.get('gibbs_risk', -1.0),
                        stats.get('tandem_risk', -1.0),
                        bounds.get('MUBernstein', -1.0),
                        mu,
                        stats.get('mutandem_risk', -1.0),
                        stats.get('vartandem_risk', -1.0),
                        stats.get('varUB', -1.0),
                        stats.get('bernTandemUB', -1.0),
                        bl,
                        bg
                            )
                    )
            f.write('\n')
        """
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk','gibbs','tandem_risk','pbkl','c1','c2','ctd','tnd','mub', 'bern','lamda','gamma','mu']]))
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
        """


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
    # define a grid of mu for 'MU' and 'MUBernstein'
    mu_grid = [(-0.1+0.01*i) for i in range(20)]
    
    print("Training...")
    _  = rf.fit(trainX,trainY)
    _, mv_risk = rf.predict(testX,testY)
    stats  = rf.stats(options = {'mu_mub': mu_grid, 'mu_bern': mu_grid}) # initial stats after training
    bounds, stats = rf.bounds(stats=stats) # compute the bounds according to the best mu in the grid, and record the corresponding stats
    res_unf = (mv_risk, stats, bounds, -1, -1, -1)
    
    # Optimize Lambda
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_mub = mu_bern = [0.0]
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_lam = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)

        
    # Optimize TND
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats) # update rho-dependent stats, and reinitialize mu_mub = mu_bern = [0.0]
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_tnd = (mv_risk, stats, bounds, bl, -1, -1)
    rhos.append(rho)
    
    """
    # Optimize MU
    print("Optimizing MU...")
    (_, rho, bl, bg, mu) = rf.optimize_rho('MU', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds = rf.bounds(stats=stats)
    res_mu = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)
    """
    
    # Optimize MU with grid
    print("Optimizing MU (using grid) in [-0.1, 0.1] ...")
    (_, rho, mu, bl, bg) = rf.optimize_rho('MU', options={'optimizer':OPT,'mu_grid':mu_grid})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_mub':[mu]}) # update rho-dependent stats, and let mu_mub = [mu], mu_bern = [0.0]
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_mug = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)
    
    # Optimize MUBernstein with grid
    print("Optimizing MUBernstein (using grid) in [-0.1, 0.1] ...")
    (_, rho, mu, bl, bg) = rf.optimize_rho('MUBernstein', options={'optimizer':OPT,'mu_grid':mu_grid})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats, options={'mu_bern':[mu]}) # update rho-dependent stats, and let mu_mub = [0.0], mu_bern = mu
    bounds, stats = rf.bounds(stats=stats) # compute the bounds and the stats with the above mus
    res_MUBernsteing = (mv_risk, stats, bounds, bl, bg, mu)
    rhos.append(rho)


    # opt = (bound, rho, lam, gam, mu)
    if rep==0:
        # record the \rho distribution by all optimization methods
        _write_dist_file('rho-'+DATASET, rhos, stats['risks'])
    results.append((rep, n, (res_unf, res_lam, res_tnd, res_mug, res_MUBernsteing)))
    
_write_outfile(results)

