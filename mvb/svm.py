#
# Implements an SVM based majority vote in the framework
#
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from . import util, mvbase

class SVMVotersClassifier(mvbase.MVBounds):
    def __init__(
            self,
            n_estimators,
            n_train, # 'dim' = d+1, int or float
            Cs=None,
            gammas=None,
            rho=None,
            random_state=None
            ):
        
        self._Cs = Cs if Cs is not None else [10**i for i in range(-3,4)]
        self._gammas = gammas if gammas is not None else [10**i for i in range(-4,5)]

        prng = check_random_state(random_state)
        
        estimators = []
        for i in range(n_estimators):
            C = prng.choice(self._Cs)
            gamma = prng.choice(self._gammas)
            svm = SVC(C=C, gamma=gamma)
            estimators.append(svm)

        super().__init__(estimators, rho, sample_mode=n_train, random_state=prng)

    def fit(self, X, Y):
        estimate = super().fit(X,Y)
        return estimate
