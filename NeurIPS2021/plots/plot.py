"""
Prepare data to show the results after running optimize.py
Requirement : Need data to be in ../out/optimize/
"""

import sys
import numpy as np
import pandas as pd
import os

BASE = sys.argv[1] if len(sys.argv)>=2 else 'rfc'
M = int(sys.argv[2]) if len(sys.argv)>=3 else 100

EXP_PATH = "../out/optimize/"+BASE+"/"
RENAME = {"Fashion-MNIST":"Fashion"}
DATASETS = [
            'SVMGuide1',
            'Phishing',
            'Mushroom',
            'Splice',
            'w1a',
            'Cod-RNA',
            'Adult',
            'Protein',
            'Connect-4',
            'Shuttle',
            'Pendigits',
            'Letter',
            'SatImage',
            'Sensorless',
            'USPS',
            'MNIST',
            'Fashion-MNIST',
            ]


"""
Prepare data for comparison of MV risk/bounds bewteen \rho=\rho* and \rho=uniform.
The data will be recorded in figure/"+base+"/datasets, where base is 'rfc' or 'mce'.
Ex. Figure 2 in the NeurIPS 2021 paper
"""
def optimized_comparison(tp='risk', base='rfc'):
    path = "figure/"+base+"/datasets/"
    if not os.path.isdir(path):
        os.makedirs(path)

    opts, baseline = {
        ("risk","rfc"):   (["lam","tnd","cctnd","ccpbb"],["unf"]),
        ("risk","mce"):   (["best","lam","tnd","cctnd","ccpbb"],["unf"]),
        ("bound","rfc"):  ([("lam","PBkl"),("tnd","TND"),("cctnd","CCTND"),("ccpbb","CCPBB")],[("tnd","TND")]),
        ("bound","mce"):  ([("lam","PBkl"),("tnd","TND"),("cctnd","CCTND"),("ccpbb","CCPBB")],[("tnd","TND")])
    }[(tp,base)]
    if tp=='risk':
        opts = [(o,"mv_risk") for o in opts]
        baseline = [(b,"mv_risk") for b in baseline]

    for bl,blbnd in baseline:
        cols = ["dataset"]
        for opt,_ in opts:
            cols += [opt+suf for suf in ["_diff","_q25","_q75"]]
        rows_bin = []
        rows_mul = []
        blcol = bl+"_"+blbnd
        for ds in DATASETS:
            if (base == 'rfc' and ds == 'Protein'):
                continue
            df = pd.read_csv(EXP_PATH+ds+"-"+str(M)+"-bootstrap-iRProp.csv",sep=";")
            if (df[blcol]==0).sum() > 0:
                continue
            row = [RENAME.get(ds,ds)]
            for opt,obnd in opts:
                optcol = opt+"_"+obnd
                diff   = df[optcol]/df[blcol]
                med = diff.median()
                row += [med, med-diff.quantile(0.25), diff.quantile(0.75)-med]
            if df["c"].iloc[0]==2:
                rows_bin.append(row)
            else:
                rows_mul.append(row)
        
        pd.DataFrame(data=rows_bin, columns=cols).to_csv(path+tp+"-bin-"+bl + ".csv", sep=";", index_label="idx")
        pd.DataFrame(data=rows_mul, columns=cols).to_csv(path+tp+"-mul-"+bl + ".csv", sep=";", index_label="idx")

optimized_comparison("risk",base=BASE)
optimized_comparison("bound",base=BASE)

PREC = 4

PRETTY_MAP = {
    "best_mv_risk":"$L(h_{best})$",
    "unf_mv_risk":"$L(\\MV_{u})$",
    "lam_mv_risk":"$L(\\MV_{\\rho_\\lambda})$",
    "tnd_mv_risk":"$L(\\MV_{\\rho_{\\TND}})$",
    "cctnd_mv_risk":"$L(\\MV_{\\rho_{\\CCTND}})$",
    "ccpbb_mv_risk":"$L(\\MV_{\\rho_{\\CCPBB}})$",
    "lam_PBkl":"$\\FO(\\rho_\\lambda)$",
    "tnd_TND":"$\\TND(\\rho_{\\TND})$",
    "cctnd_CCTND":"$\\CCTND(\\rho_{\\CCTND})$",
    "ccpbb_CCPBB":"$\\CCPBB(\\rho_{\\CCPBB})$",
    "lam":"$\\FO$",
    "tnd":"$\\TND$",
    "cctnd":"$\\CCTND$",
    "ccpbb":"$\\CCPBB$",
    "gibbs":"$\\E_\\rho[L]$",
    "tandem":"$\\E_{\\rho^2}[L]$",
    "bmu":"$\mu$",
}

"""
Making tables for comparison of MV risk/bounds bewteen \rho=\rho* and \rho=uniform.
The .tex files will be recorded in table/"+base+"/optimize/, where base is 'rfc' or 'mce'.
Ex. Table 2,3,5,6 in the NeurIPS 2021 paper
"""
def optimized_comparison_table(tp='risk', base='rfc', hl1="all", hl2=[]):
    path = "table/"+base+"/optimize/"
    out_fname = path+tp+"_table.tex"
    if not os.path.isdir(path):
        os.makedirs(path)

    opts = {
        ("risk","rfc"):       ["unf","lam","tnd","cctnd","ccpbb"],
        ("risk","mce"):       ["unf","best","lam","tnd","cctnd","ccpbb"],
        ("bound","rfc"):      [("lam","PBkl"),("tnd","TND"),("cctnd","CCTND"),("ccpbb","CCPBB")],
        ("bound","mce"):      [("lam","PBkl"),("tnd","TND"),("cctnd","CCTND"),("ccpbb","CCPBB")],
    }[(tp,base)]
    if tp=='risk':
        opts = [(o,"mv_risk") for o in opts]
    
    copts = [pre+"_"+suf for pre,suf in opts]
    
    if hl1=="all":
        hl1 = copts

    with open(out_fname, 'w') as fout:
        # Header
        fout.write("\\begin{tabular}{l"+"c"*len(opts)+"}\\toprule\n")
        fout.write("Data set")
        for i,col in enumerate(copts):
            fout.write(" & "+PRETTY_MAP[col])
        fout.write(" \\\\\n")
        fout.write("\\midrule\n")

        for ds in DATASETS:
            if (base == 'rfc' and ds == 'Protein'):
                continue
            df = pd.read_csv(EXP_PATH+ds+"-"+str(M)+"-bootstrap-iRProp.csv",sep=";")
            df_mean = df.mean()
            df_std  = df.std()
            
            # Highlight indices
            v1 = np.min(df_mean[hl1]) if len(hl1)>0 else -1
            v2 = np.min(df_mean[hl2]) if len(hl2)>0 else -1
            v1 = str(round(v1,PREC))
            v2 = str(round(v2,PREC))

            fout.write("\\dataset{"+RENAME.get(ds,ds)+"}")
            for i,col in enumerate(copts):
                fval = df_mean[col]
                val = str(round(fval,PREC))
                std = str(round(df_std[col],PREC))
                s = val + " ("+std+")"
                if col in hl1 and val==v1:
                    s = "\\textbf{"+s+"}"
                if col in hl2 and val==v2:
                    s = "\\underline{"+s+"}"
                fout.write(" & "+s)
            fout.write(" \\\\\n")

        fout.write("\\bottomrule\n") 
        fout.write("\\end{tabular}\n")
    
optimized_comparison_table('risk', base=BASE, hl2=["lam_mv_risk","tnd_mv_risk","cctnd_mv_risk","ccpbb_mv_risk"])
optimized_comparison_table('bound', base=BASE, hl2=["tnd_TND","cctnd_CCTND","ccpbb_CCPBB"])


"""
Making tables with various values:
bmu, gibbs loss and tandem loss for different optimization methods.
The .tex files will be recorded in table/"+base+"/optimize/, where base is 'rfc' or 'mce'.
Ex. Table 4,7 in the NeurIPS 2021 paper
"""
def optimized_values_table(base="rfc"):
    path = "table/"+base+"/optimize/"
    out_fname = path+"values_table.tex"
    if not os.path.isdir(path):
        os.makedirs(path)
    
    opts = ["lam","tnd","cctnd","ccpbb"]
    cols = []
    for o in opts:
        cols += [o+"_"+suf for suf in ["gibbs","tandem"]]
        if o in ("cctnd","ccpbb"):
            cols.append(o+"_bmu")
    
    with open(out_fname, 'w') as fout:
        # Header
        fout.write("\\begin{tabular}{l"+"c"*len(cols)+"}\\toprule\n")
        for o in opts:
            cnt = "3" if o in ("cctnd","ccpbb") else "2"
            fout.write(" & \multicolumn{"+cnt+"}{|c|}{"+PRETTY_MAP.get(o,o)+"}")
        fout.write(" \\\\\n")
        fout.write("Data set")
        for i,col in enumerate(cols):
            _,cn = col.split("_")
            bl = "|" if i in (0,2,4,7) else ""
            br = "|" if i==len(cols)-1 else ""
            fout.write(" & \multicolumn{1}{"+bl+"c"+br+"}{"+PRETTY_MAP.get(cn,cn)+"}")
        fout.write(" \\\\\n")
        fout.write("\\midrule\n")

        for ds in DATASETS:
            if (base == 'rfc' and ds == 'Protein'):
                continue
            df = pd.read_csv(EXP_PATH+ds+"-"+str(M)+"-bootstrap-iRProp.csv",sep=";")
            df_mean = df.mean()
            df_std  = df.std()
            
            fout.write("\\dataset{"+RENAME.get(ds,ds)+"}")
            for i,col in enumerate(cols):
                fval = df_mean[col]
                val = str(round(fval,PREC))
                std = str(round(df_std[col],PREC))
                s = val# + " ("+std+")"
                fout.write(" & "+s)
            fout.write(" \\\\\n")

        fout.write("\\bottomrule\n") 
        fout.write("\\end{tabular}\n")

optimized_values_table(base=BASE)

# NOT USED
def TND_Ben_comparison_table(base='rfc'):
    path = "table/"+base+"/optimize/"
    if not os.path.isdir(path):
        os.makedirs(path)
    
    prec = 5
    opts = ["tnd", "mu", "bern"]
    cols = ["dataset", "c", "d"]
    for opt in opts:
        if opt == "tnd":
            cols += [opt+suf for suf in ["_KL", "_gibbs", "_tandem", "_tnd", "_TandemUB"]]
        elif opt == "mu":
            cols += [opt+suf for suf in ["_KL", "_gibbs", "_tandem", "_MU", "_muTandemUB", "_bmu"]]
        elif opt == "bern":
            cols += [opt+suf for suf in ["_KL", "_gibbs", "_tandem", "_bern", '_mutandem_risk', '_vartandem_risk', "_varUB", "_bernTandemUB", "_bmu", "_bg", "_bl"]]
    rows = []
    for ds in DATASETS:
        if (base == 'rfc' and ds == 'Protein'):
            continue
        df = pd.read_csv(EXP_PATH+ds+"-"+str(M)+"-bootstrap-iRProp.csv",sep=";")
        df_mean = df.mean()
        df_std  = df.std()
        
        row = [ds, df_mean["c"], df_mean["d"]]
        for opt in opts:
            if opt == "tnd":
                row += [df_mean[opt+suf] for suf in ["_KL", "_gibbs", "_tandem", "_tnd", "_TandemUB"]]
            elif opt == "mu":
                row += [df_mean[opt+suf] for suf in ["_KL", "_gibbs", "_tandem", "_MU", "_muTandemUB", "_bmu"]]
            elif opt == "bern":
                row += [df_mean[opt+suf] for suf in ["_KL", "_gibbs", "_tandem", "_bern", '_mutandem_risk', '_vartandem_risk', "_varUB", "_bernTandemUB", "_bmu", "_bg", "_bl"]]            
        rows.append(row)
    
    pd.DataFrame(data=rows, columns=cols).round(prec).to_csv(path+"mu_comparison.csv", sep=",", index=False)

#TND_Ben_comparison_table(base=BASE)
