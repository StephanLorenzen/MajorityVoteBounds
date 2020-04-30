import sys
import os
import numpy as np

from rfb import RandomForestWithBounds as RFWB
from rfb import data as mldata

inpath  = 'data/'
outpath = 'out/uniform/'

# tree strength, set None for full
opts = set(sys.argv)
weak = '-w' in opts or '--weak' in opts

DATA_SETS = [
        ('Splice',      200),
        ('Adult',       200),
        ('GermanNumer', 200),
        ('SVMGuide1',   200),
        ('w1a',         200),
        ('Phishing',    200),
        #('Madelon',     200),
        ('Letter:AB',   200),
        ('Letter:DO',   200),
        ('Letter:OQ',   200),
        ('Mushroom',    200),
        ]
DATA_SETS.sort(key=lambda x: x[0])
    
max_depth = 2 if weak else None
max_features= 1 if weak else None

bag_results    = [[], []]
val_results    = [[], []]
bagval_results = [[], []]

print("Starting tests...")
if(weak):
    print(" -> Using decision stumps")
else:
    print(" -> Using full trees")


for dataset,m in DATA_SETS:
    print("##### "+dataset+" #####")

    X,Y = mldata.load(dataset, path=inpath)
    
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8)
    
    print("Training forest for ["+dataset+"] with bagging")
    rf = RFWB(n_estimators=m,max_depth=max_depth,max_features=max_features)
    _  = rf.fit(trainX,trainY)
    _, mv_risk = rf.predict(testX,testY)
    n = (trainX.shape[0], 0, testX.shape[0])
    stats  = rf.stats()
    bounds = rf.bounds(stats=stats)
    bag_results[0].append((mv_risk, n, bounds, stats))
     
    stats  = rf.stats(unlabeled_data=testX)
    bounds = rf.bounds(unlabeled_data=testX, stats=stats)
    bag_results[1].append((mv_risk, n, bounds, stats))
    
    # Further split train X and Y
    trainX,trainY,valX,valY = mldata.split(trainX,trainY,0.5)
    print("Training forest for ["+dataset+"] with bagging and validation set")
    rf = RFWB(n_estimators=m,max_depth=max_depth,max_features=max_features)
    _ = rf.fit(trainX,trainY)
    _, mv_risk = rf.predict(testX,testY)
    n = (trainX.shape[0], valX.shape[0], testX.shape[0])
    stats  = rf.stats(val_data=(valX,valY))
    bounds = rf.bounds(val_data=(valX,valY), stats=stats)
    bagval_results[0].append((mv_risk, n, bounds, stats))

    stats  = rf.stats(val_data=(valX,valY), unlabeled_data=testX)
    bounds = rf.bounds(val_data=(valX,valY), unlabeled_data=testX, stats=stats)
    bagval_results[1].append((mv_risk, n, bounds, stats))
    
    # No need to retrain for val set only
    print("Computing bounds on validation set only")
    n = (trainX.shape[0], valX.shape[0], testX.shape[0])
    stats  = rf.stats(val_data=(valX,valY), incl_oob=False)
    bounds = rf.bounds(val_data=(valX,valY), incl_oob=False, stats=stats)
    val_results[0].append((mv_risk, n, bounds, stats))
    
    stats  = rf.stats(val_data=(valX,valY), unlabeled_data=testX, incl_oob=False)
    bounds = rf.bounds(val_data=(valX,valY), unlabeled_data=testX, incl_oob=False, stats=stats)
    val_results[1].append((mv_risk, n, bounds, stats))
    

if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(name, results):
    prec = 5
    name = ("ds-" if weak else "rf-")+name
    with open(outpath+name+'.csv', 'w') as f:
        f.write('dataset;n_train;n_val;n_test;m;mv_risk;gibbs;disagreement;u_disagreement;tandem_risk;pbkl;c1;c2;mv2;mv2u;sh;best_tree;worst_tree\n')
        for (ds,m), (risk_mv, n, bounds, stats) in zip(DATA_SETS, results):
            f.write(ds+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(m)+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(13)])+'\n')
                    .format(risk_mv,
                        stats['risk'],
                        stats['disagreement'],
                        stats.get('u_disagreement', 1.0),
                        stats['tandem_risk'],
                        bounds['PBkl'],
                        bounds['C1'],
                        bounds['C2'],
                        bounds['MV2'],
                        bounds.get('MV2u',1.0),
                        bounds.get('SH',1.0),
                        np.min(stats['risks']),
                        np.max(stats['risks']))
                    )

_write_outfile('bag-results', bag_results[0])
_write_outfile('bag-ul-results', bag_results[1])
_write_outfile('val-results', val_results[0])
_write_outfile('val-ul-results', val_results[1])
_write_outfile('bagval-results', bagval_results[0])
_write_outfile('bagval-ul-results', bagval_results[1])

