from mvb import RandomForestClassifier as RF
from mvb import data as mldata

dataset = 'Letter:AB'
m = 100

print("Loading data set ["+dataset+"]...")
X, Y = mldata.load(dataset)
print("Done!")

print("\n######### Bagging #########")
print("Fitting random forest...")
rf = RF(n_estimators=m)
oob_estimate = rf.fit(X, Y)
print("Done! OOB estimate: "+str(oob_estimate))
print("")
print("Computing statistics and bounds...")
stats  = rf.stats()
bounds = rf.bounds(stats=stats)
print("Gibbs risk: "+str(stats['gibbs_risk']))
print("Disagreement: "+str(stats['disagreement']))
print("Tandem risk: "+str(stats['tandem_risk']))
print("Bounds:")
print("  PBkl: "+str(bounds['PBkl']))
print("  C1:   "+str(bounds['C1']))
print("  C2:   "+str(bounds['C2']))
print("  MV:   "+str(bounds['MV']))

print("\n######### Bagging+Validation #########")
X, Y, v_X, v_Y = mldata.split(X, Y, 0.5)
print("Fitting random forest...")
oob_estimate = rf.fit(X, Y)
print("Done! OOB estimate: "+str(oob_estimate))
print("Predicting on validation data...")
_, risk_mv = rf.predict(v_X, v_Y)
print("Done! MV risk on val: "+str(risk_mv))
print("")
print("Computing statistics and bounds...")
stats  = rf.stats(labeled_data=(v_X,v_Y))
bounds = rf.bounds(labeled_data=(v_X,v_Y), stats=stats)
print("Gibbs risk: "+str(stats['gibbs_risk']))
print("Disagreement: "+str(stats['disagreement']))
print("Tandem risk: "+str(stats['tandem_risk']))
print("Bounds:")
print("  PBkl: "+str(bounds['PBkl']))
print("  C1:   "+str(bounds['C1']))
print("  C2:   "+str(bounds['C2']))
print("  MV:   "+str(bounds['MV']))
print("  SH:   "+str(bounds['SH']))


