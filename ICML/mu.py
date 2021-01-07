#
# Runs experiments using standard random forest with uniform weighting.
# Computes the MU bound for various values of mu \in [0,0.5)
#
# Usage: python uniform.py <data_set> [M] [sample_mode] [repeats]
#        M           : number of trees
#        sample_mode : 'bootstrap', 'dim', f in [0,1]
#                      'bootstrap' = full bagging.
#                      'dim'       = sample d points with replacement
#                      float f     = sample f*|S| points with replacement
#        repeats     : integer
#
# Return: results saved in out
#
import sys
import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import data as mldata

inpath  = 'data/'
outpath = 'out/mu/'

SEED = 1000

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
SMODE   = sys.argv[3] if len(sys.argv)>=4 else 'bootstrap'
SMODE   = SMODE if (SMODE=='bootstrap' or SMODE=='dim') else float(SMODE)
REPS    = int(sys.argv[4]) if len(sys.argv)>=5 else 1

num_vals = 100
mu_vals  = [(i/num_vals-0.5) for i in range(num_vals)]

RAND = check_random_state(SEED)
if not os.path.exists(outpath):
    os.makedirs(outpath)

smodename = 'bagging' if SMODE=='bootstrap' else ('reduced bagging ('+str(SMODE)+');')
print("Starting RFC experiment (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename)
columns = ["n_train","n_test","d","c","mv_risk","gibbs_risk","n_min","disagreement","tandem_risk","n2_min","pbkl","tnd"]+["mub_"+str(mu) for mu in mu_vals]
results = dict([(c,[]) for c in columns])
for rep in range(REPS):
    if REPS>1:
        print("##### Repeat = "+str(rep))
    print("Loading data...")
    X,Y = mldata.load(DATASET, path=inpath)
    C = np.unique(Y).shape[0]
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
    
    print("Training...")
    rf = RFC(n_estimators=M, max_features="sqrt", random_state=RAND,sample_mode=SMODE)
    _  = rf.fit(trainX,trainY)
    
    print("Evaluating final classifier...")
    _, mv_risk = rf.predict(testX,testY)

    print("Computing bounds...")
    stats  = rf.stats()
    # Hack: nothing needed for estimating mu, so use everything for computing MU bound
    stats["r_tandem_risk"] = stats["tandem_risk"]
    stats["r_gibbs_risk"]  = stats["gibbs_risk"]
    stats["r_n_min"]       = stats["n_min"]
    stats["r_n2_min"]      = stats["n2_min"]
    
    results["n_train"].append(trainX.shape[0])
    results["n_test"].append(testX.shape[0])
    results["d"].append(trainX.shape[1])
    results["c"].append(C)
    results["mv_risk"].append(mv_risk)
    for col in ["gibbs_risk","n_min","disagreement","tandem_risk","n2_min"]:
        results[col].append(stats[col])

    # Bounds
    results["pbkl"].append(rf.bound("PBkl", stats=stats))
    results["tnd"].append(rf.bound("TND", stats=stats))
    for mu in mu_vals:
        stats["mu"] = mu
        results["mub_"+str(mu)].append(rf.bound("MU", stats=stats))

pd.DataFrame(results).to_csv(outpath+DATASET+".csv", index_label="repeat")

