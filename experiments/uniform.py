import sys
import os
import numpy as np
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import ExtraTreesClassifier as ETC
from mvb import data as mldata

inpath  = 'data/'
outpath = 'out/uniform/'

SEED = 1000

DATA_SETS = [
        ('Splice',      100),
        ('Adult',       100),
        ('SVMGuide1',   100),
        ('w1a',         100),
        ('Phishing',    100),
        #('Letter:AB',   200),
        #('Letter:DO',   200),
        #('Letter:OQ',   200),
        ('Mushroom',    100),
        ('Letter',      100),
        ('Shuttle',     100),
        #('Segment',     200),
        ('Pendigits',   100),
        ('Protein',     100),
        ('SatImage',    100),
        ('Sensorless',  100),
        ('USPS',        100),
        ('Connect-4',   100),
        ('Cod-RNA',     100),
        ('mnist',       100)
        ]
DATA_SETS.sort(key=lambda x: x[0])
    
print("Starting tests...")
RAND = check_random_state(SEED)
if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(name, results):
    prec = 5
    with open(outpath+name+'.csv', 'w') as f:
        f.write('dataset;n_train;n_test;d;c;m;mv_risk;gibbs;n_min;disagreement;u_disagreement;tandem_risk;n2_min;pbkl;c1;c2;mv2;mv2u;sh;best_tree;worst_tree\n')
        for (ds,m), (risk_mv, n, bounds, stats) in zip(DATA_SETS, results):
            f.write(ds+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+str(m)+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(15)])+'\n')
                    .format(risk_mv,
                        stats['gibbs_risk'],
                        stats['n_min'],
                        stats['disagreement'],
                        stats.get('u_disagreement', -1.0),
                        stats['tandem_risk'],
                        stats['n2_min'],
                        bounds['PBkl'],
                        bounds.get('C1', -1.0),
                        bounds.get('C2', -1.0),
                        bounds['MV'],
                        bounds.get('MVu',-1.0),
                        bounds.get('SH',-1.0),
                        np.min(stats['risks']),
                        np.max(stats['risks']))
                    )


for rep in range(50):
    print("##### Repeat = "+str(rep))
    results = []
    for dataset,m in DATA_SETS:
        print("Training RFC for ["+dataset+"] with bagging")
        X,Y = mldata.load(dataset, path=inpath)
        C = np.unique(Y).shape[0]
    
        trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
        n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)
        
        rf = RFC(n_estimators=m, max_features="sqrt",random_state=RAND)
        _  = rf.fit(trainX,trainY)
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        results.append((mv_risk, n, bounds, stats))
    
    suffix = str(rep) if rep > 9 else ('0'+str(rep))
    _write_outfile('results-rep-'+suffix, results)

