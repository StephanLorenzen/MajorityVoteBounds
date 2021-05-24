#
# Implements AdaBoostClassifier in the framework
#
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from math import ceil

from .bounds import SH
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
            random_state=None,
            n_splits = 2, # split the samples into n_splits splits and build an AdaBoost for each split
            use_ada_prior=False
            ):
        self._actual_n_estimators = n_estimators
        dt_stump = Tree(max_depth = max_depth, 
                        min_samples_leaf = min_samples_leaf)
                        
        prng = check_random_state(random_state)
        
        estimators = [None] * n_estimators
        if n_splits == 2:
            abc = [AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators, algorithm=algorithm, random_state=prng)]
        else:
            # divide m classifiers into n_splits groups, (m/n_splits , m/n_splits , remaining) 
            m = n_estimators
            m_group = [int(m/n_splits) for i in range(n_splits-1)]
            m_group.append(m-sum(m_group))
            
            abc = [AdaBoostClassifier(base_estimator=dt_stump, n_estimators=m_group[i], algorithm=algorithm, random_state=prng) for i in range(n_splits)]
        
        super().__init__(estimators, abc, rho, sample_mode=sample_mode, random_state=prng, use_ada_prior=use_ada_prior)
    
    # prepare the base classifiers and validation data for PAC-Bayes methods
    def fit(self, X, Y):
        estimate = super().fit(X,Y)
        return estimate
    
    # return the number of estimators
    def get_n_estimators(self):
        return self._actual_n_estimators


class BaseAdaBoostClassifier():
    def __init__(
            self,
            n_estimators,
            min_samples_leaf=1,
            max_depth = 1, # -> decision stump
            algorithm='SAMME',
            random_state=None,
            n_splits = 2, # split the samples into n_splits splits and build an AdaBoost for each split
            ):

        self.n_splits = n_splits
        self.n_estimators= n_estimators
        self._prng       = check_random_state(random_state)

        dt_stump = Tree(max_depth = max_depth, 
                        min_samples_leaf = min_samples_leaf)
                        
        self.abc = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators, algorithm=algorithm, random_state=self._prng)
           
    # prepare the base classifiers and validation data for PAC-Bayes methods
    def fit(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        self._classes = np.unique(Y)
        
        # use 1./n_splits of the data for training
        t_X,val_X,t_Y,val_Y = train_test_split(X, Y, train_size= 1./self.n_splits, shuffle=True, stratify=Y, random_state=self._prng)
        self.VAL = (val_X,val_Y)
        """
        n = X.shape[0]
        n_sample = ceil(1./self.n_splits * n)
        
        # sample points for training (wo. replacement)
        while True: 
            # Repeat until at least one example of each class
            t_idx = self._prng.choice(n, size=n_sample, replace=False)
            t_X = X[t_idx]
            t_Y = Y[t_idx]
            if np.unique(t_Y).shape[0] == self._classes.shape[0]:
                break

        # validation samples
        val_idx = np.delete(np.arange(n),t_idx)
        val_X   = X[val_idx]
        val_Y   = Y[val_idx]
        self.VAL = (val_X,val_Y)
        """
        # fit
        self.abc.fit(t_X, t_Y)
        
        # calculate rho
        rho = self.abc.estimator_weights_
        rho = rho/np.sum(rho)
        return rho
    
    # return the number of estimators
    def get_n_estimators(self):
        return self._actual_n_estimators
    
    # predict and calculate the risk
    def predict(self, X, Y):
        return self.abc.predict(X), 1.0 - self.abc.score(X, Y)
        
    # calculate the SH bound on validation data
    def bound(self):
        stats = dict()
        bounds = dict()
        
        X, Y = self.VAL
        _, stats['mv_risk'] = self.predict(X, Y)
        stats['mv_n'] = len(self.VAL[1])
        
        bounds['SH'] = SH(stats['mv_risk'], stats['mv_n'])
        return bounds, stats