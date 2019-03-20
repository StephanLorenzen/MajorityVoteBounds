import sys
import os

from rfb import RandomForestWithBounds as RFWB
from rfb import data as mldata

inpath  = 'data/'
outpath = 'out/'

DATA_SETS = [
        ('Letter:AB',   200),
        ('Letter:DO',   200),
        ('Letter:OQ',   200),
        ('Tic-Tac-Toe', 200),
        ('Sonar',       50),
        ('USVotes',     50),
        ('WDBC',        100),
        ('Heart',       50),
        ('Haberman',    50),
        ('Ionosphere',  50),
        ('ILPD',        100),
        ('Mushroom',    1000),
        ('Credit-A',    100)
        ]

bag_results    = []
val_results    = []
bagval_results = []
for dataset,m in DATA_SETS:
    print("##### "+dataset+" #####")
    
    X,Y = mldata.load(dataset, path=inpath)

    trainX,trainY,testX,testY = mldata.split(X,Y,0.5)
    
    print("Training forest for ["+dataset+"] with bagging")
    rf = RFWB(n_estimators=m)
    _, bounds = rf.fit(trainX,trainY)
    _, _, details = rf.predict_all(testX,testY,return_details=True)
    bag_results.append((details['risk_mv'], bounds))

    # Further split train X and Y
    trainX,trainY,valX,valY = mldata.split(trainX,trainY,0.5)
    print("Training forest for ["+dataset+"] with bagging and validation set")
    rf = RFWB(n_estimators=m)
    _, bounds = rf.fit(trainX,trainY,val_X=valX,val_Y=valY)
    _, _, details = rf.predict_all(testX,testY,return_details=True)
    bagval_results.append((details['risk_mv'], bounds))

    # No need to retrain for val set only
    print("Computing bounds on validation set only")
    _, bounds = rf.predict_all(valX,valY)
    val_results.append((details['risk_mv'], bounds))
    
if not os.path.exists(outpath):
    os.makedirs(outpath)
def _write_outfile(name, results):
    prec = 5
    with open(outpath+name+'.csv', 'w') as f:
        f.write('dataset;m;risk_mv;pbkl;c1;c2;sh\n')
        for (ds,m), (risk_mv, bounds) in zip(DATA_SETS, results):
            f.write(ds+';'+str(m)+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(5)])+'\n')
                    .format(risk_mv, bounds['PBkl'], bounds['C1'], bounds['C2'], bounds['SH'])
                    )

_write_outfile('bag-results', bag_results)
_write_outfile('val-results', val_results)
_write_outfile('bagval-results', bagval_results)

