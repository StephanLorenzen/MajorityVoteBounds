#
# Runs sparsification experiments.
#
# Usage: python sparsify.py <data_set> [M]
#        M           : number of initial trees
#
# Return: results saved in out
#
import sys
import os
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state

from mvb import RandomForestClassifier as RFC
from mvb import data as mldata

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100

inpath  = 'data/'
outpath = 'out/sparsify/'

SEED = 1000

if not os.path.exists(outpath):
    os.makedirs(outpath)

print("Starting RFC sparsification (m = "+str(M)+") for ["+DATASET+"]")

results = {"M":[]}
for approach in ["OOB","Lambda","TND"]:
    results[approach+"_mvrisk"] = []
    results[approach+"_unf_mvrisk"] = []
    results[approach+"_weakest"] = []
    results[approach+"_FO"] = []
    results[approach+"_TND"] = []

X,Y = mldata.load(DATASET, path=inpath)
C = np.unique(Y).shape[0]

for method in ["OOB", "Lambda", "TND"]:
    rf = RFC(M,max_features="sqrt",random_state=SEED, sample_mode="bootstrap")
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=SEED)
    
    print("Training...")
    _  = rf.fit(trainX,trainY)
   
    use_oob = False
    if method=="OOB":
        use_oob=True
    else:
        rf.optimize_rho(method, options={"optimizer":"iRProp"})

    risks, ns = rf.risks()
    s = 1.0-np.max(risks/ns) if use_oob else np.min(rf._rho)
    r = 0
    print("Removing based on: "+str(method))
    for c in range(M):
        _, mv_risk = rf.predict(testX,testY)
        if c%10==0:
            print("Removed "+str(c)+", risk = "+str(round(mv_risk,3))+", worst voter left: "+str(round(s,4)))
        if method=="OOB":
            results["M"].append(len(rf._estimators))
        results[method+"_mvrisk"].append(mv_risk)
        results[method+"_weakest"].append(s)
        results[method+"_FO"].append(rf.bound("PBkl"))
        results[method+"_TND"].append(rf.bound("TND"))

        if method=="OOB":
            results[method+"_unf_mvrisk"].append(mv_risk)
        else:
            # Small hack to get uniform risk
            m = len(rf._rho)
            old_rho = rf._rho
            rf._rho = np.ones(m)/m
            _, unf_mv_risk = rf.predict(testX,testY)
            results[method+"_unf_mvrisk"].append(unf_mv_risk)
            rf._rho = old_rho
        
        # Sparsify
        r, s = rf.sparsify(n=1, use_oob=use_oob)

pd.DataFrame(results).set_index("M").to_csv(outpath+DATASET+".csv",index_label="M")
