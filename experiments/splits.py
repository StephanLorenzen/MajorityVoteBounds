import sys
import os
import numpy as np
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import ExtraTreesClassifier as ETC
from mvb import data as mldata

inpath  = 'data/'
outpath = 'out/splits/'

SEED = 1000

DATA_SETS = [
        ('Splice',      200),
        ##('Adult',       200),
        ##('GermanNumer', 200),
        ('SVMGuide1',   200),
        ##('w1a',         200),
        ##('Phishing',    200),
        ##('Letter',      200),
        ('Letter:AB',   200),
        ('Letter:DO',   200),
        ('Letter:OQ',   200),
        ('Mushroom',    200),
        ##('Shuttle',     200),
        ##('Segment',     200),
        ##('Pendigits',   200)
        ]
DATA_SETS.sort(key=lambda x: x[0])

max_depth=None #2
max_features="sqrt" #1
results = {}

print("Starting tests...")
RAND = check_random_state(SEED)

for dataset,m in DATA_SETS:
    print("##### "+dataset+" #####")
    X,Y = mldata.load(dataset, path=inpath)
    C = np.unique(Y).shape[0]
    print("n =",X.shape[0],"d =",X.shape[1],"#classes =",C)
    print("") 

    results[dataset] = []
    for train_split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        trainX,trainY,testX,testY = mldata.split(X,Y,train_split,random_state=RAND)
        n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)
         
        print("Training RFC for ["+dataset+"], bagging, split = "+str(train_split))
        rf = RFC(n_estimators=m, max_depth=max_depth, max_features=max_features, random_state=RAND)
        _  = rf.fit(trainX,trainY)
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        results[dataset].append((train_split, mv_risk, m, n, bounds, stats))
    
    print("") 


if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(name, results):
    prec = 5
    with open(outpath+name+'.csv', 'w') as f:
        f.write('n_train;n_test;d;c;m;split;mv_risk;gibbs;disagreement;u_disagreement;tandem_risk;pbkl;c1;c2;mv2;mv2u;sh;best_tree;worst_tree\n')
        for (split, risk_mv, m, n, bounds, stats) in results:
            f.write(str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+str(m)+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(14)])+'\n')
                    .format(split,
                        risk_mv,
                        stats['risk'],
                        stats['disagreement'],
                        stats.get('u_disagreement', -1.0),
                        stats['tandem_risk'],
                        bounds['PBkl'],
                        bounds.get('C1', -1.0),
                        bounds.get('C2', -1.0),
                        bounds['MV2'],
                        bounds.get('MV2u',-1.0),
                        bounds.get('SH',-1.0),
                        np.min(stats['risks']),
                        np.max(stats['risks']))
                    )

for (ds, res) in results.items():
    _write_outfile(ds, res)

