import libsl.ml.data.mldata as mldata
from rfb import RandomForestWithBounds as RFWB

dataset = 'letter'
m = 100

print("Loading data set [Letter:AB]...")
X, Y = mldata.load(dataset,permutate=True,cond=('A','B'))
print("Done!")

print("\n######### Bagging #########")
print("Fitting random forest and computing bounds...")
rf = RFWB(n_estimators=m)
oob_estimate, bounds, details = rf.fit(X, Y, return_details=True)
print("Done!")
print("")

print("OOB estimate: "+str(oob_estimate))
print("Gibbs risk: "+str(details['risk_gibbs']))
print("Disagreement: "+str(details['disagreement']))
print("Bounds:")
print("  PBkl: "+str(bounds['PBkl']))
print("  C1:   "+str(bounds['C1']))
print("  C2:   "+str(bounds['C2']))

print("\n######### Bagging+Validation #########")
X, Y, v_X, v_Y = mldata.load(dataset,permutate=True,split=0.5,cond=('A','B'))
print("Fitting random forest and computing bounds...")
rf = RFWB(n_estimators=m)
risk_mv, bounds, details = rf.fit(X, Y, val_X=v_X, val_Y=v_Y, return_details=True)
print("Done!")
print("")

print("MV risk on val: "+str(risk_mv))
print("Gibbs risk: "+str(details['risk_gibbs']))
print("Disagreement: "+str(details['disagreement']))
print("Bounds:")
print("  PBkl: "+str(bounds['PBkl']))
print("  C1:   "+str(bounds['C1']))
print("  C2:   "+str(bounds['C2']))
print("  SH:   "+str(bounds['SH']))


