import numpy as np
import pandas as pd
import os

DATASETS = [
        'SVMGuide1',
        'Phishing',
        'Mushroom',
        'Pendigits',
        'Splice',
        'Adult',
        'w1a',
        'Letter',
        'Shuttle',
        'Protein',
        'SatImage',
        'Sensorless',
        'USPS',
        'Connect-4',
        'Cod-RNA',
        'MNIST',
        'Fashion-MNIST',
        ]
EXP_PATH  = "../out/"
NUM_TREES = 100
BOUNDS_BINARY = [("pbkl","FO"),("c1","Cone"),("c2","Ctwo"),("ctd","CTD"),("tnd","TND"),("mub","MU")]
BOUNDS_MULTI  = BOUNDS_BINARY[:1]+BOUNDS_BINARY[3:]

# Plot error and bounds for several data sets
def multi_bounds(exp="uniform"):
    name = "bounds_"+exp
    path = name+"/datasets/"
    if not os.path.isdir(path):
        os.makedirs(path)

    for ds in DATASETS:
        df = pd.read_csv(EXP_PATH+exp+"/"+ds+"-"+str(NUM_TREES)+"-bootstrap.csv",sep=";")
        df_mean = df.mean()
        df_std  = df.std()**(0.5)
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
 

# Prep data files for mu_plots
DATASETS = [
        'Phishing',
        'Mushroom',
        'Pendigits',
        'Letter',
        'Sensorless',
        'Cod-RNA',
        ]

def mu_plot():
    name = "mu_plot"
    path = name+"/datasets/"
    if not os.path.isdir(path):
        os.makedirs(path)
   
    stats = {"dataset":DATASETS,
            "mv_risk_mean":[],"mv_risk_std":[],
            "risk_mean":[],"risk_std":[],
            "tandem_mean":[],"tandem_std":[]}
    for ds in DATASETS:
        df = pd.read_csv(EXP_PATH+"mu/"+ds+".csv")
        # Compute means and stds
        df_mean = df.mean()
        df_std  = df.std()
        output = {"mu":[],"risk_mean":[],"risk_std":[],"pbkl_mean":[],"pbkl_std":[],"tnd_mean":[],"tnd_std":[],"mu_mean":[],"mu_std":[]}
        for c in df.columns:
            if c[:4]=="mub_":
                output["mu"].append(float(c[4:]))
                output["risk_mean"].append(df_mean["mv_risk"])
                output["risk_std"].append(df_std["mv_risk"])
                output["mu_mean"].append(df_mean[c])
                output["mu_std"].append(df_std[c])
                for cc in ["pbkl","tnd"]:
                    output[cc+"_mean"].append(df_mean[cc])
                    output[cc+"_std"].append(df_std[cc])
        
        stats["mv_risk_mean"].append(df_mean["mv_risk"])
        stats["mv_risk_std"].append(df_std["mv_risk"])
        stats["risk_mean"].append(df_mean["gibbs_risk"])
        stats["risk_std"].append(df_std["gibbs_risk"])
        stats["tandem_mean"].append(df_mean["tandem_risk"])
        stats["tandem_std"].append(df_std["tandem_risk"])

        pd.DataFrame(output).to_csv(path+ds+".csv", index_label="idx")
    pd.DataFrame(stats).to_csv(path+"stats.csv", index_label="idx")

mu_plot()
