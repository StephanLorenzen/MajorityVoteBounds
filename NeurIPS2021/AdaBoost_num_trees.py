#
# Purpose: Test the test_risk v.s. number of trees for the typical AdaBoost classifier in our datasets
#
# Create date : 21 May, 2021
#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

from mvb import BaseAdaBoostClassifier as baseABC
from sklearn.ensemble import AdaBoostClassifier
from mvb import data as mldata

DATASETS = [
        'SVMGuide1',
        'Phishing',
        'Mushroom',
        'Splice',
        'w1a',
        'Cod-RNA',
        'Adult',
        'Connect-4',
        'Shuttle',
        ]

inpath  = 'data/'
SEED = 1000
RAND = check_random_state(SEED)
M = 200
SPLITS = 3
REP = 3

def run():
    for ds in DATASETS:
        outpath = 'AdaBoost-test/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        result = []
        # Loading data
        print("Loading data...", ds)
        X,Y = mldata.load(ds, path=inpath)
        
        for rep in range(REP):
            print('rep', rep)
            trainX,trainY,testX,testY = mldata.split(X,Y,0.8,random_state=RAND)
            trainX,trainY,valX,valY = mldata.split(trainX,trainY,2/3,random_state=RAND)
            
            test_risk = []
            for m in range(0, M):
                abc = AdaBoostClassifier(n_estimators=m+1, algorithm='SAMME', random_state=RAND)
                abc.fit(trainX, trainY)
                risk = 1. - abc.score(testX, testY)
                test_risk.append(risk)
            result.append(test_risk)
        
        pd.DataFrame(result).to_csv(outpath+ds+'.csv', index=False)
#run()

def plot():
    for ds in DATASETS:
        # read data
        inpath = 'AdaBoost-test/'+ds+'.csv'
        df = pd.read_csv(inpath)
        mean = df.mean().transpose()
        x_idx = np.arange(1,len(mean)+1)
        
        # plot
        fig = plt.figure()
        plt.title('test_risk v.s. number of trees')
        plt.ylabel('test_risk')
        plt.xlabel('trees')
        plt.plot(x_idx, mean, label=ds)
        plt.legend()
        plt.savefig('AdaBoost-test/'+ds+'.png')
plot()
