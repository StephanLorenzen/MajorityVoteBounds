import sys
import os
import numpy as np
from sklearn.utils import check_random_state


from mvb import RandomForestClassifier as RFC
from mvb import ExtraTreesClassifier as ETC
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
m = 50

rf_results = []
ef_results = []
ds_results = []

def _write_dist_file(name, rhos, risks):
    with open(outpath+name+'.csv', 'w') as f:
        f.write("h;risk;rho_lam;rho_mv2;rho_mv2u\n")
        for i,(err,r_lam,r_mv,r_mvu) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_mv)+";"+str(r_mvu)+"\n")

print("Starting tests...")
if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

for name, rf, reslist in [("rf", RFC(m,max_features="sqrt",random_state=RAND), rf_results)]:#, ("ef", ETC(m,max_features="sqrt",random_state=RAND), ef_results), ("ds", RFC(m,max_features=1,max_depth=2,random_state=RAND), ds_results)]:
    print("Starting "+name+" experiment")
    for dataset in DATA_SETS:
        print("##### "+dataset+" #####")
        X,Y = mldata.load(dataset, path=inpath)
        C = np.unique(Y).shape[0]
        print("n =",X.shape[0],"d =",X.shape[1],"#classes =",C)
        print("") 

        trainX,trainY,testX,testY = mldata.split(X,Y,0.5,random_state=RAND)
        n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

        rhos = []
        print("Training RFC for ["+dataset+"] with bagging")
        _  = rf.fit(trainX,trainY)
        print(" => rho = uniform")
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_unf = (mv_risk, stats, bounds)
        print(" => rho = rho_lambda")
        (_, rho, _) = rf.optimize_rho('Lambda')
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_lam = (mv_risk, stats, bounds)
        rhos.append(rho)
        print(" => rho = rho_mv")
        (_, rho, _) = rf.optimize_rho('MV2')
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats(unlabeled_data=testX)
        bounds = rf.bounds(unlabeled_data=testX, stats=stats)
        res_mv2 = (mv_risk, stats, bounds)
        rhos.append(rho)
        if(C==2):
            print(" => rho = rho_mvu")
            (_, rho, _, _) = rf.optimize_rho('MV2',unlabeled_data=testX)
            _, mv_risk = rf.predict(testX,testY)
            stats  = rf.stats(unlabeled_data=testX)
            bounds = rf.bounds(unlabeled_data=testX, stats=stats)
            res_mv2u = (mv_risk, stats, bounds)
            rhos.append(rho)
        else:
            res_mv2u = (-1.0, dict(), dict())
            rhos.append(-np.ones((m,)))

        # opt = (bound, rho, lam, gam)
        _write_dist_file(name+"-"+dataset, rhos, stats['risks'])
        reslist.append((res_unf, res_lam, res_mv2, res_mv2u))

        print("") 

def _write_outfile(name, results):
    prec = 5
    with open(outpath+name+'.csv', 'w') as f:
        f.write('dataset;n_train;n_test;d;c;m')
        for name in ["unf","lam","mv2","mv2u"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk','gibbs','disagreement','u_disagreement','tandem_risk','pbkl','c1','c2','mv2','mv2u']]))
        f.write('\n')
        m = 100
        for ds, restup in zip(DATA_SETS, results):
            f.write(ds+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+str(m));
            for (mv_risk, stats, bounds) in restup:
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(10)]))
                        .format(mv_risk,
                            stats.get('risk', -1.0),
                            stats.get('disagreement', -1.0),
                            stats.get('u_disagreement', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('C1', -1.0),
                            bounds.get('C2', -1.0),
                            bounds.get('MV2', -1.0),
                            bounds.get('MV2u',1.0)
                            )
                        )
            f.write('\n')

_write_outfile('rf-results', rf_results)
_write_outfile('ef-results', ef_results)
_write_outfile('ds-results', ds_results)
