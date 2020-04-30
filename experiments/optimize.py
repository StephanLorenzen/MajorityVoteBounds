import sys
import os

from rfb import RandomForestWithBounds as RFWB
from rfb import data as mldata

inpath  = 'data/'
outpath = 'out/optimize/'

# tree strength, set None for full
opts = set(sys.argv)
weak = '-w' in opts or '--weak' in opts

DATA_SETS = [
        ('Splice',      200),
        #('Adult',       200),
        #('GermanNumer', 200),
        #('SVMGuide1',   200),
        #('w1a',         200),
        #('Phishing',    200),
        #('Madelon',     200),
        ('Letter:AB',   200),
        #('Letter:DO',   200),
        #('Letter:OQ',   200),
        #('Mushroom',    200),
        ]
DATA_SETS.sort(key=lambda x: x[0])
    
max_depth = 2 if weak else None
prefix = "ds-" if weak else "rf-"

bag_results    = [[], [], []]
bagval_results = [[], [], []]

print("Starting optimization tests...")
if(weak):
    print(" -> Using decision stumps")
else:
    print(" -> Using full trees")

def _write_dist_file(name, rho, risks):
    with open(outpath+prefix+name+'.csv', 'w') as f:
        f.write("risk;rho\n")
        for (r,err) in zip(rho,risks):
            f.write(str(err)+";"+str(r)+"\n")

for dataset,m in DATA_SETS:
    print("##### "+dataset+" #####")

    X,Y = mldata.load(dataset, path=inpath)
    
    trainX,trainY,testX,testY = mldata.split(X,Y,0.8)
    
    print("Training forest for ["+dataset+"] with bagging")
    n = (trainX.shape[0], 0, testX.shape[0])
    rf = RFWB(n_estimators=m,max_depth=max_depth)
    _  = rf.fit(trainX,trainY)
    
    (mv2, mv2_rho, mv2_lam) = rf.optimize_rho('MV2')
    _, mv2_mv_risk = rf.predict(testX,testY)
    mv2_stats  = rf.stats()
    mv2_bounds = rf.bounds(stats=mv2_stats)
    bag_results[0].append((mv2_mv_risk, n, mv2_bounds, mv2_stats, mv2_lam, 0.0))
    _write_dist_file("bag-dist-mv2-"+dataset, mv2_rho, mv2_stats['risks'])

    (u_mv2, u_mv2_rho, u_mv2_lam, u_mv2_gam) = rf.optimize_rho('MV2',unlabeled_data=testX)
    _, u_mv2_mv_risk = rf.predict(testX,testY)
    u_mv2_stats  = rf.stats(unlabeled_data=testX)
    u_mv2_bounds = rf.bounds(unlabeled_data=testX, stats=u_mv2_stats)
    bag_results[1].append((u_mv2_mv_risk, n, u_mv2_bounds, u_mv2_stats, u_mv2_lam, u_mv2_gam))
    _write_dist_file("bag-dist-umv2-"+dataset, u_mv2_rho, u_mv2_stats['risks'])

    (lam, lam_rho, lam_lam) = rf.optimize_rho('Lambda')
    _, lam_mv_risk = rf.predict(testX,testY)
    lam_stats  = rf.stats()
    lam_bounds = rf.bounds(stats=lam_stats)
    bag_results[2].append((lam_mv_risk, n, lam_bounds, lam_stats, lam_lam, 0.0))
    _write_dist_file("bag-dist-lam-"+dataset, lam_rho, lam_stats['risks'])

    # Further split train X and Y
    trainX,trainY,valX,valY = mldata.split(trainX,trainY,0.5)
    print("Training forest for ["+dataset+"] with bagging and validation set")
    rf = RFWB(n_estimators=m,max_depth=max_depth)
    _ = rf.fit(trainX,trainY)
    n = (trainX.shape[0], valX.shape[0], testX.shape[0])
    
    vald = (valX, valY)
    (mv2, mv2_rho, mv2_lam) = rf.optimize_rho('MV2', val_data=vald)
    _, mv2_mv_risk = rf.predict(testX,testY)
    mv2_stats  = rf.stats(val_data=vald)
    mv2_bounds = rf.bounds(val_data=vald, stats=mv2_stats)
    bagval_results[0].append((mv2_mv_risk, n, mv2_bounds, mv2_stats, mv2_lam, 0.0))
    _write_dist_file("bagval-dist-mv2-"+dataset, mv2_rho, mv2_stats['risks'])
    
    (u_mv2, u_mv2_rho, u_mv2_lam, u_mv2_gam) = rf.optimize_rho('MV2', val_data=vald, unlabeled_data=testX)
    _, u_mv2_mv_risk = rf.predict(testX,testY)
    u_mv2_stats  = rf.stats(val_data=vald, unlabeled_data=testX)
    u_mv2_bounds = rf.bounds(val_data=vald, unlabeled_data=testX, stats=u_mv2_stats)
    bagval_results[1].append((u_mv2_mv_risk, n, u_mv2_bounds, u_mv2_stats, u_mv2_lam, u_mv2_gam))
    _write_dist_file("bagval-dist-umv2-"+dataset, u_mv2_rho, u_mv2_stats['risks'])

    (lam, lam_rho, lam_lam) = rf.optimize_rho('Lambda', val_data=vald)
    _, lam_mv_risk = rf.predict(testX,testY)
    lam_stats  = rf.stats(val_data=vald)
    lam_bounds = rf.bounds(val_data=vald, stats=lam_stats)
    bagval_results[2].append((lam_mv_risk, n, lam_bounds, lam_stats, lam_lam, 0.0))
    _write_dist_file("bagval-dist-lam-"+dataset, lam_rho, lam_stats['risks'])
    
if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(name, results):
    prec = 5
    with open(outpath+prefix+name+'.csv', 'w') as f:
        f.write('dataset;n_train;n_val;n_test;m;mv_risk;gibbs;disagreement;u_disagreement;tandem_risk;pbkl;c1;c2;mv2;mv2u;sh;lam;gam\n')
        for (ds,m), (risk_mv, n, bounds, stats, lam, gam) in zip(DATA_SETS, results):
            f.write(ds+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(m)+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(13)])+'\n')
                    .format(risk_mv,
                        stats['risk'],
                        stats['disagreement'],
                        stats.get('u_disagreement', -1),
                        stats['tandem_risk'],
                        bounds['PBkl'],
                        bounds['C1'],
                        bounds['C2'],
                        bounds['MV2'],
                        bounds.get('MV2u', -1),
                        bounds.get('SH', -1),
                        lam,
                        gam)
                    )

_write_outfile('bag-mv2-results', bag_results[0])
_write_outfile('bag-umv2-results', bag_results[1])
_write_outfile('bag-lam-results', bag_results[2])
_write_outfile('bagval-mv2-results', bagval_results[0])
_write_outfile('bagval-umv2-results', bagval_results[1])
_write_outfile('bagval-lam-results', bagval_results[2])

