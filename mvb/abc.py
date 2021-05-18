#
# Implements AdaBoostClassifier in the framework
#
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import check_random_state

from . import util, mvbase

class OurAdaBoostClassifier(mvbase.MVBounds):
    def __init__(
            self,
            n_estimators,
            rho=None,
            min_samples_leaf=1,
            max_depth = 1, # -> decision stump
            sample_mode="boost",
            algorithm='SAMME',
            random_state=None
            ):
        self._actual_n_estimators = n_estimators
        dt_stump = Tree(max_depth = max_depth, 
                        min_samples_leaf = min_samples_leaf)
                        
        prng = check_random_state(random_state)
        
        estimators = [None] * n_estimators        
        abc = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators, algorithm=algorithm, random_state=prng)
        
        super().__init__(estimators, abc, rho, sample_mode=sample_mode, random_state=prng)
    
    # prepare the base classifiers and validation data for PAC-Bayes methods
    def fit(self, X, Y):
        estimate = super().fit(X,Y)
        return estimate
    
    # return the number of estimators
    def get_n_estimators(self):
        return self._actual_n_estimators
    
    # risk of the typical AdaBoost method
    def adaboost_risk(self, X, Y):
        return 1.0 - self._ensembled_estimators.score(X, Y)
