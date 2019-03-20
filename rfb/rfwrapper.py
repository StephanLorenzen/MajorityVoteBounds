import random
import numpy as np

from . import util
from .bounds import SH, PBkl, C1, C2, C3

class RandomForestWithBounds:
    def __init__(
            self,
            n_estimators,
            rho=None,
            criterion="gini",
            max_features=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            max_depth=None,
            seed=None,
            lib='sklearn-rfc' # or sklearn-erc or woody
            ):
        self._trees            = []
        self._bootstrap        = bootstrap
        self._max_depth        = max_depth
        self._actual_max_depth = max_depth
       
        self._rho = rho
        if rho is None:
            self._rho = (1.0/n_estimators)*np.ones((n_estimators,))
        assert(self._rho.shape[0] == n_estimators)

        if lib not in ['sklearn-rfc', 'sklearn-etc', 'woody']:
            util.error('Unknown lib: "'+str(lib)+'".')
        self._lib = lib

        if lib == 'woody':
            from woody import WoodClassifier as Tree
            for i in range(n_estimators):
                tree = Tree(
                        n_estimators=1,
                        criterion=criterion,
                        max_features=max_features,
                        min_samples_split=min_samples_split,
                        n_jobs=1,
                        bootstrap=False,
                        tree_traversal_mode="dfs",
                        tree_type="standard",
                        min_samples_leaf=min_samples_leaf,
                        float_type="double",
                        max_depth=max_depth,
                        verbose=0)
            self._trees.append(tree)
        else:
            if lib == 'sklearn-rfc':
                from sklearn.ensemble import RandomForestClassifier as Tree
            else:
                from sklearn.ensemble import ExtraTreesClassifier as Tree
            for i in range(n_estimators):
                tree = Tree(
                        n_estimators=1,
                        criterion=criterion,
                        max_features=max_features,
                        min_samples_split=min_samples_split,
                        n_jobs=1,
                        bootstrap=False,
                        min_samples_leaf=min_samples_leaf,
                        max_depth=max_depth,
                        verbose=0)
                self._trees.append(tree)


    def fit(self, X, Y, val_X=None, val_Y=None, rho=None, return_details=False):
        
        validation = not (val_X is None or val_Y is None)
        
        # No bootstrapping or validation sets
        if not (self._bootstrap or validation):
            # No bounds will be computed
            util.warn(
            "Warning, RandomForestWithBounds.fit: not possible to compute \
                    generalization bounds, when not using bootstrapping and \
                    no validation sets have been supplied."
                    )
            
            for tree in self._trees:
                tree.fit(X, Y)
            self._actual_max_depth = max([t.estimators_[0].tree_.max_depth for t in self._trees])
            return None
        
        else:
            oob_preds = []
            val_preds = []

            X, Y = np.array(X), np.array(Y)
            if validation:
                val_X, val_Y = np.array(val_X), np.array(val_Y)
            
            n = X.shape[0]
            m = len(self._trees)
            for tree in self._trees:
                t_X, t_Y = X, Y
                oob_idx, oob_X = None, None
                
                if self._bootstrap:
                    # Sample points for training (w. replacement)
                    t_idx = np.random.randint(n, size=n)
                    t_X   = X[t_idx]
                    t_Y   = Y[t_idx]

                    # OOB sample
                    oob_idx = np.delete(np.arange(n),np.unique(t_idx))
                    oob_X   = X[oob_idx]

                # Fit this tree
                tree.fit(t_X, t_Y)

                # OOB / Validation
                oob_P, val_P = None, None
                if self._bootstrap:
                    # Predict on OOB
                    oob_P = tree.predict(oob_X)
                if validation:
                    # Predict on validation set
                    val_P = tree.predict(val_X)

                # Save predictions on oob and validation set for later
                oob_preds.append((oob_idx, oob_P)) 
                val_preds.append(val_P)

            if self._lib != 'woody':
                self._actual_max_depth = max([t.estimators_[0].tree_.max_depth for t in self._trees])
            
            oob_set, val_set = None, None
            emp_risk = 0.0
            if self._bootstrap:
                oob_set = (oob_preds, Y)
                emp_risk = util.compute_oob_estimate(self._rho, oob_set)
            if validation:
                val_preds = np.array(val_preds)
                val_set  = (val_preds, val_Y)
                emp_risk = util.compute_mv_risk(self._rho, val_preds, val_Y)
            
            # Compute stats:
            stats = util.compute_stats(oob_set, val_set, self._rho)
            
            # Compute bounds
            pi = util.uniform_distribution(len(self._trees))
            KL = util.compute_kl(self._rho, pi)

            pbkl = PBkl(stats['risk_gibbs'], stats['n_min'], KL)
            c1 = C1(stats['risk_gibbs'], stats['disagreement'], stats['n_min'], stats['jn_min'], KL)
            c2 = C2(stats['joint_error'], stats['disagreement'], stats['jn_min'], KL)
            sh = SH(stats['risk_mv'], val_X.shape[0]) if validation else 1.0

            bounds = {
                    "PBkl":pbkl,
                    "C1":c1,
                    "C2":c2,
                    "SH":sh
                    }
            
            stats['n_val'] = val_X.shape[0] if validation else 0
            return (emp_risk, bounds, stats) if return_details else (emp_risk, bounds)

    def predict(self, X):
        n = X.shape[0]

        P = self.predict_all(X)
        return util.compute_mv_preds(self._rho, P)

    def predict_all(self, X, Y=None, return_details=False):
        n = X.shape[0]
        m = len(self._trees)
        
        P = []
        for tree in self._trees:
            P.append(tree.predict(X))
        
        P = np.array(P)

        if Y is None:
            return P
        
        else:
            # Compute bounds
            stats = util.compute_stats(None, (P, Y), self._rho)

            # Compute bounds
            pi = util.uniform_distribution(len(self._trees))
            KL = util.compute_kl(self._rho, pi)

            pbkl = PBkl(stats['risk_gibbs'], stats['n_min'], KL)
            c1 = C1(stats['risk_gibbs'], stats['disagreement'], stats['n_min'], stats['jn_min'], KL)
            c2 = C2(stats['joint_error'], stats['disagreement'], stats['jn_min'], KL)
            sh = SH(stats['risk_mv'], X.shape[0])

            bounds = {
                    "PBkl":pbkl,
                    "C1":c1,
                    "C2":c2,
                    "SH":sh
                    }

            stats['n'] = X.shape[0]

            return (P, bounds, stats) if return_details else (P, bounds)

    def get_max_depth(self):
        if self._lib == 'woody':
            util.warn(
            'Warning, RandomForestWithBounds.get_max_depth: \
            Actual max depth not available when using "Woody", \
            paramter "max_depth" returned.'
            )
        return self.actual_depth
