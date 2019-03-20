from rfb import RandomForestWithBounds as RFWB
from rfb import data as mldata

DATA_SETS = [
        ('Letter:AB',100),
        ('Letter:DO',100),
        ('Letter:OQ',100),
        ('ILPD',     100),
        ('Mushroom', 100),
        ('Credit-A', 100)]

bag_results    = []
val_results    = []
bagval_results = []
for dataset,m in DATA_SETS:
    print("##### "+dataset+" #####")
    
    X,Y = mldata.load(dataset)

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
    

