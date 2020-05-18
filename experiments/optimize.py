import sys
import os
import numpy as np
import numpy.linalg as LA
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import data as mldata

inpath  = 'data/'
outpath = 'out/optimize/'

SEED = 1000

DATA_SETS = [
        'Splice',
        'Adult',
        'w1a',
        'SVMGuide1',
        'Phishing',
        'Letter:AB', 
        'Letter:DO',
        'Letter:OQ',
        'Mushroom',
        'Letter',
        'Segment',
        'Shuttle',
        'Pendigits'
        ]
DATA_SETS.sort(key=lambda x: x[0])
m = 100

def _write_dist_file(name, rhos, risks):
    with open(outpath+name+'.csv', 'w') as f:
        f.write("h;risk;rho_lam;rho_mv2;rho_mv2u\n")
        for i,(err,r_lam,r_mv,r_mvu) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_mv)+";"+str(r_mvu)+"\n")

print("Starting tests...")
if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

def _write_outfile(name, results):
    prec = 5
    with open(outpath+name+'.csv', 'w') as f:
        f.write('dataset;n_train;n_test;d;c;m')
        for name in ["unf","lam","mv2","mv2u"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk','gibbs','disagreement','u_disagreement','tandem_risk','pbkl','c1','c2','mv2','mv2u','lamda','gamma']]))
        f.write('\n')
        m = 100
        for ds, (n, restup) in zip(DATA_SETS, results):
            f.write(ds+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+str(m));
            for (mv_risk, stats, bounds, bl, bg) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(12)]))
                        .format(mv_risk,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('disagreement', -1.0),
                            stats.get('u_disagreement', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('C1', -1.0),
                            bounds.get('C2', -1.0),
                            bounds.get('MV', -1.0),
                            bounds.get('MVu',1.0),
                            bl,
                            bg
                            )
                        )
            f.write('\n')

for rep in range(50):
    print("####### Repeat = "+str(rep))
    rf = RFC(m,max_features="sqrt",random_state=RAND)
    reslist = []
    for dataset in DATA_SETS:
        print("Training RFC for ["+dataset+"] with bagging")
        X,Y = mldata.load(dataset, path=inpath)
        C = np.unique(Y).shape[0]
    
        trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
        n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)
    
        rhos = []
        _  = rf.fit(trainX,trainY)
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
    
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_unf = (mv_risk, stats, bounds, -1, -1)
        
        # Optimize Lambda
        (_, rho, bl) = rf.optimize_rho('Lambda')
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_lam = (mv_risk, stats, bounds, bl, -1)
        rhos.append(rho)
        
        # Optimize MV
        (_, rho, bl) = rf.optimize_rho('MV')
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_mv2 = (mv_risk, stats, bounds, bl, -1)
        rhos.append(rho)

        # Optimize MVu if binary
        if(C==2):
            (_, rho, bl, bg) = rf.optimize_rho('MV',unlabeled_data=testX)
            _, mv_risk = rf.predict(testX,testY)
            stats = rf.aggregate_stats(stats)
            bounds = rf.bounds(unlabeled_data=testX, stats=stats)
            res_mv2u = (mv_risk, stats, bounds, bl, bg)
            rhos.append(rho)
        else:
            res_mv2u = (-1.0, dict(), dict(), -1, -1)
            rhos.append(-np.ones((m,)))
    
        # opt = (bound, rho, lam, gam)
        if rep==0:
            _write_dist_file('rho-'+dataset, rhos, stats['risks'])
        reslist.append((n, (res_unf, res_lam, res_mv2, res_mv2u)))
    
    suffix = ('0' if rep < 10 else '')+str(rep)
    _write_outfile('results-rep-'+suffix, reslist)

