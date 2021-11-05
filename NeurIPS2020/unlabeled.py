#
# Runs experiments with unlabeled data.
#
# Usage: python unlabeled.py <data_set> [M] [frac] [repeats]
#        M       : number of trees
#        frac    : f in [0,1], fraction of S used for training, the remainder is used unlabeled
#        repeats : integer
#
# Return: results saved in out
#
import sys
import os
import numpy as np
from sklearn.utils import check_random_state

from mvb import data as mldata
from mvb import RandomForestClassifier as RFC

inpath  = 'data/'
outpath = 'out/unlabeled/'

SEED = 1000

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
TFRAC   = float(sys.argv[3]) if len(sys.argv)>=4 else 0.1
REPS    = int(sys.argv[4]) if len(sys.argv)>=5 else 1

RAND = check_random_state(SEED)
if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(results):
    prec = 5
    with open(outpath+DATASET+'-'+str(M)+'-'+str(TFRAC)+'.csv', 'w') as f:
        f.write('repeat;n_train;n_ulb;n_test;d;c;mv_risk;gibbs;n_min;disagreement;disagreement_t;tandem_risk;n2_min;n2_min_t;pbkl;c1;c2;tnd;ctd;dis;sh;best_tree;worst_tree\n')
        for (rep, risk_mv, n, bounds, stats) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+str(n[4])+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(17)])+'\n')
                    .format(risk_mv,
                        stats['gibbs_risk'],
                        stats['n_min'],
                        stats['disagreement'],
                        stats.get('u_disagreement', -1.0),
                        stats['tandem_risk'],
                        stats['n2_min'],
                        stats['u_n2_min'],
                        bounds['PBkl'],
                        bounds.get('C1', -1.0),
                        bounds.get('C2', -1.0),
                        bounds['TND'],
                        bounds['CTD'],
                        bounds.get('DIS',-1.0),
                        bounds.get('SH',-1.0),
                        np.min(stats['risks']),
                        np.max(stats['risks']))
                    )

print("Starting RFC experiment (m = "+str(M)+") for ["+DATASET+"] using bagging and "+str(TFRAC)+"N training data")
results = []
print("Loading data...")
X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]
for rep in range(REPS):
    if REPS>1:
        print("##### Repeat = "+str(rep))
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    trainX,trainY,ulX,  _     = mldata.split(trainX,trainY,TFRAC,random_state=RAND)  
    n = (trainX.shape[0], ulX.shape[0], testX.shape[0], trainX.shape[1], C)
    
    print("Training...")
    rf = RFC(n_estimators=M, max_features='sqrt',random_state=RAND,sample_mode='bootstrap')
    _  = rf.fit(trainX,trainY)
    
    print("Evaluating...")
    _, mv_risk = rf.predict(testX,testY)
    
    print("Computing bounds...")
    stats  = rf.stats(unlabeled_data=ulX)
    bounds, stats = rf.bounds(stats=stats)
    results.append((rep, mv_risk, n, bounds, stats))
    
_write_outfile(results)

