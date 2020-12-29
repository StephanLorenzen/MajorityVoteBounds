import numpy as np
import pandas as pd
import os

DATASETS = [
        ('SVMGuide1','bin'),
        ('Phishing','bin'),
        ('Mushroom','bin'),
        ('Pendigits','mul'),
        ('Splice','bin'),
        ('Adult','bin'),
        ('w1a','bin'),
        ('Letter','bin'),
        ('Shuttle','bin'),
        ('Protein','bin'),
        ('SatImage','bin'),
        ('Sensorless','bin'),
        ('USPS','bin'),
        ('Connect-4','bin'),
        ('Cod-RNA','bin'),
        ('MNIST','mul'),
        ('Fashion-MNIST','mul'),
        ]
EXP_PATH  = "../out/"
NUM_TREES = 100
BOUNDS_BINARY = [("pbkl","FO"),("c1","Cone"),("c2","Ctwo"),("ctd","CTD"),("tnd","TND"),("mub","MU")]
BOUNDS_MULTI  = BOUNDS_BINARY[:1]+BOUNDS_BINARY[3:]

def multi_bounds(exp="uniform"):
    name = "bounds_"+exp
    path = name+"/datasets/"
    os.makedirs(path)

    for ds,b in DATASETS:
        df = pd.read_csv(EXP_PATH+exp+"/"+ds+"-"+str(NUM_TREES)+"-bootstrap.csv",sep=";")
        df_mean = df.mean()
        df_std  = df.std()
        bounds = BOUNDS_BINARY if df_mean["c"]==2 else BOUNDS_MULTI
        with open(path+ds+".tex", "w") as f:
            for i,(bnd,cls) in enumerate(bounds):
                f.write("\\addplot["+cls+", Bound]coordinates {("+str(i+1)+","+str(df_mean[bnd])+") +- (0,"+str(df_std[bnd])+")};\n")
                
            mean, err = df_mean['mv_risk'], df_std['mv_risk']
            up   = str(mean+err)
            lw   = str(mean-err)
            mean = str(mean)
            f.write("\\addplot+[RiskErr,name path=UP] {"+up+"};\n")
            f.write("\\addplot+[RiskErr,name path=LW] {"+lw+"};\n")
            f.write("\\addplot[RiskErr] fill between[of=UP and LW];\n")
            f.write("\\addplot[Risk] coordinates {(0,"+mean+") (7,"+mean+")};\n")

multi_bounds()
 

