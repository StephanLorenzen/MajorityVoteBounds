from rfb import RandomForestWithBounds as RFWB
from rfb import data as mldata

dataset = 'Letter:AB'
m = 100

print("Loading data set ["+dataset+"]...")
X, Y = mldata.load(dataset)
import pdb; pdb.set_trace()
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
X, Y, v_X, v_Y = mldata.split(X, Y, 0.5)
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


