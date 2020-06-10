#
# Implements the main framework. Majority vote classifiers should implement this
# framework by inheriting.
#
# Takes care of bagging/sample modes, computing statistics and bounds, and
# optimizing weights.
#
# Implemented to fit the signature of similar classifiers from sklearn
#
import numpy as np
from sklearn.utils import check_random_state

from . import util
from .bounds import SH, PBkl, optimizeLamb, C1, C2, CTD, TND, optimizeTND, DIS, optimizeDIS
from math import ceil

class MVBounds:
    # Constructor
    def __init__(
            self,
            estimators,
            rho=None,
            sample_mode=None, # | 'bootstrap' | 'dim' | int | float
            random_state=None,
            ):
        self._estimators = estimators
        m                = len(estimators)  
        self._sample_mode= sample_mode
        self._prng       = check_random_state(random_state)
        self._rho = rho
        if rho is None:
            self._rho = util.uniform_distribution(m)
        assert(self._rho.shape[0]==m)

        # Some fitting stats
        self._OOB = None
        self._classes = None

    # Fitting procedure
    def fit(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        self._classes = np.unique(Y)
        self._rho = util.uniform_distribution(len(self._estimators))

        # No sampling
        if self._sample_mode is None:
            self._OOB = None

            for est in self._estimators:
                est.fit(X, Y)
            return None
        
        else:
            preds = []
            n = X.shape[0]
            m = len(self._estimators)
            
            n_sample = None
            if self._sample_mode=='bootstrap':
                n_sample = n
            elif self._sample_mode=='dim':
                n_sample = X.shape[1]+1
            elif type(self._sample_mode) is int:
                n_sample = self._sample_mode
            elif type(self._sample_mode) is float:
                n_sample = ceil(n*self._sample_mode)
            else:
                Utils.warn('Warning, fit: unknown sample_type')
                return None

            for est in self._estimators:
                oob_idx, oob_X = None, None
                # Sample points for training (w. replacement)
                while True: 
                    # Repeat until at least one example of each class
                    t_idx = self._prng.randint(n, size=n_sample)
                    t_X   = X[t_idx]
                    t_Y   = Y[t_idx]
                    if np.unique(t_Y).shape[0] > 1:
                        break

                # OOB sample
                oob_idx = np.delete(np.arange(n),np.unique(t_idx))
                oob_X   = X[oob_idx]

                # Fit this estimator
                est.fit(t_X, t_Y)
                # Predict on OOB
                oob_P = est.predict(oob_X)

                M_est, P_est = np.zeros(Y.shape), np.zeros(Y.shape)
                M_est[oob_idx] = 1
                P_est[oob_idx] = oob_P

                # Save predictions on oob and validation set for later
                preds.append((M_est,P_est))
            
            self._OOB = (preds, Y)
            return self.risk()

    # Predict for an input list. If Y!=None, returns risk on (X,Y) as well.
    def predict(self, X, Y=None):
        n = X.shape[0]

        P = self.predict_all(X)
        mvP = util.mv_preds(self._rho, P)
        return (mvP, util.risk(mvP, Y)) if Y is not None else mvP
        

    # Returns individual predictions for each tree.
    def predict_all(self, X):
        n = X.shape[0]
        m = len(self._estimators)
        
        P = []
        for est in self._estimators:
            P.append(est.predict(X))
        
        return np.array(P)

    # Optimizes the weights.
    def optimize_rho(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, options=None):
        if bound not in {"Lambda", "TND", "DIS"}:
            util.warn('Warning, optimize_rho: unknown bound!')
            return None
        if labeled_data is None and not incl_oob:
            util.warn('Warning, stats: Missing data!')
            return None

        if(bound=='Lambda'):
            risks, ns = self.risks(labeled_data, incl_oob)
            (bound, rho, lam) = optimizeLamb(risks/ns, np.min(ns))
            self._rho = rho
            return (bound,rho,lam)
        elif(bound=='TND'):
            tand, n2s = self.tandem_risks(labeled_data, incl_oob)
            (bound,rho,lam) = optimizeTND(tand/n2s, np.min(n2s), options=options)
            self._rho = rho
            return (bound, rho, lam)
        else:
            ulX = unlabeled_data
            if labeled_data is not None:
                ulX = labeled_data[0] if ulX is None else np.concatenate((ulX,labeled_data[0]), axis=0)
            risks, ns = self.risks(labeled_data, incl_oob)
            dis, n2s = self.disagreements(ulX, incl_oob)
            (bound,rho,lam,gam) = optimizeDIS(risks/ns,dis/n2s,np.min(ns),np.min(n2s),options=options)
            self._rho = rho
            return (bound, rho, lam, gam)

    # Computes the given bound ('SH', 'PBkl', 'C1', 'C2', 'CTD', 'TND', 'DIS').
    # A stats object or the relevant data must be given as input (unless classifier trained
    # with bagging, in which case this data can be used).
    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, stats=None):
        if bound not in ['SH', 'PBkl', 'C1', 'C2', 'CTD', 'TND', 'DIS']:
            util.warn('Warning, MVBase.bound: Unknown bound!')
            return 1.0
        elif bound=='SH' and labeled_data==None and (stats==None or 'mv_risk' not in stats):
            util.warn('Warning, MVBase.bound: Cannot apply SH without hold-out data!')
            return 1.0
        elif bound in ['C1','C2','DIS'] and self._classes.shape[0] > 2:
            util.warn('Warning, MVBase.bound: Cannot apply '+bound+' to non-binary data!')
            return 1.0

        pi = util.uniform_distribution(len(self._estimators))
        KL = util.kl(self._rho, pi)
        if stats is not None:
            if bound=='SH':
                return SH(stats['mv_risk'], stats['mv_n'])
            elif bound=='PBkl':
                return PBkl(stats['gibbs_risk'], stats['n_min'], KL)
            elif bound=='C1':
                return C1(stats['gibbs_risk'], stats['disagreement'], stats['n_min'], stats['n2_min'], KL)
            elif bound=='C2':
                return C2(stats['tandem_risk'], stats['disagreement'], stats['n2_min'], KL)
            elif bound=='CTD':
                return CTD(stats['gibbs_risk'], stats['tandem_risk'], stats['n_min'], stats['n2_min'], KL)
            elif bound=='TND':
                return TND(stats['tandem_risk'], stats['n2_min'], KL)
            elif bound=='DIS':
                return DIS(stats['gibbs_risk'], stats['u_disagreement'], stats['n_min'], stats['u_n2_min'], KL)
            else:
                return None
        else:
            if bound=='SH':
                mv_risk, n = self.risk(labeled_data), labeled_data[0].shape[0]
                return SH(mv_risk, n)
            elif bound=='PBkl':
                grisk, n_min = self.gibbs_risk(labeled_data, incl_oob)
                return PBkl(grisk, n_min, KL)
            elif bound=='C1':
                grisk, n_min = self.gibbs_risk(labeled_data, incl_oob)
                ulX = None if labeled_data is None else labeled_data[0]
                dis, n2_min  = self.disagreement(ulX, incl_oob)
                return C1(grisk, dis, n_min, n2_min, KL)
            elif bound=='C2':
                tand, n2t = self.tandem_risk(labeled_data, incl_oob)
                ulX = None if labeled_data is None else labeled_data[0]
                dis, n2_min  = self.disagreement(ulX, incl_oob)
                assert(n2t==n2_min)
                return C2(tand, dis, n2_min, KL)
            elif bound=='CTD':
                grisk, n_min = self.gibbs_risk(labeled_data, incl_oob)
                tand, n2_min = self.tandem_risk(labeled_data, incl_oob)
                return CTD(grisk, tand, n_min, n2_min)
            elif bound=='TND':
                tand, n2_min = self.tandem_risk(labeled_data, incl_oob)
                return TND(tand, n2_min, KL)
            elif bound=='DIS':
                grisk, n_min = self.gibbs_risk(labeled_data, incl_oob)
                ulX = labeled_data[0] if labeled_data is not None else None
                if unlabeled_data is not None:
                    ulX = unlabeled_data if ulX is None else np.concat((ulX,unlabeled_data), axis=0)
                dis, n2_min  = self.disagreement(ulX, incl_oob)
                return DIS(grisk, dis, n_min, n2_min, KL)
            else:
                return None
    
    # Compute all bounds, given relevant stats object or data
    def bounds(self, labeled_data=None, unlabeled_data=None, incl_oob=True, stats=None):
        results = dict()
        if stats is None:
            stats = self.stats(labeled_data, unlabeled_data, incl_oob)

        results['PBkl'] = self.bound('PBkl', stats=stats)
        results['TND']  = self.bound('TND', stats=stats)
        results['CTD']  = self.bound('CTD', stats=stats)
        if labeled_data is not None or (stats is not None and 'mv_risk' in stats):
            results['SH'] = self.bound('SH', stats=stats)
        if self._classes.shape[0]==2:
            results['C1']  = self.bound('C1', stats=stats)
            results['C2']  = self.bound('C2', stats=stats)
            results['DIS'] = self.bound('DIS', stats=stats)
        return results
    
    # Compute stats object
    def stats(self, labeled_data=None, unlabeled_data=None, incl_oob=True):
        stats = dict()
        if labeled_data is not None:
            stats['mv_preds'] = (self.predict_all(labeled_data[0]), labeled_data[1])
            stats['mv_n']    = labeled_data[0].shape[0]

        stats['risks'], stats['n'] = self.risks(labeled_data, incl_oob)
        stats['risks'] /= stats['n']

        stats['tandem_risks'], stats['n2'] = self.tandem_risks(labeled_data, incl_oob)
        stats['tandem_risks'] /= stats['n2']

        dis, _ = self.disagreements(labeled_data[0] if labeled_data!=None else None, incl_oob)
        stats['disagreements'] = dis/stats['n2']
        if unlabeled_data is not None:
            udis, un2 = self.disagreements(unlabeled_data, False)
            stats['u_n2'] = stats['n2']+un2
            stats['u_disagreements'] = (dis+udis) / stats['u_n2']
        else:
            stats['u_n2'] = stats['n2']
            stats['u_disagreements'] = dis / stats['u_n2']
        return self.aggregate_stats(stats)
    
    # (Re-)Aggregate stats object. Useful if weighting has changed.
    def aggregate_stats(self, stats):
        if 'mv_preds' in stats:
            stats['mv_risk'] = util.mv_risk(self._rho, stats['mv_preds'][0], stats['mv_preds'][1]) 
        
        stats['gibbs_risk'] = np.average(stats['risks'], weights=self._rho)
        stats['n_min'] = np.min(stats['n'])

        stats['tandem_risk'] = np.average(np.average(stats['tandem_risks'], weights=self._rho, axis=0), weights=self._rho)
        stats['disagreement'] = np.average(np.average(stats['disagreements'], weights=self._rho, axis=0), weights=self._rho)
        stats['n2_min'] = np.min(stats['n2'])
       
        stats['u_disagreement'] = np.average(np.average(stats['u_disagreements'], weights=self._rho, axis=0), weights=self._rho)
        stats['u_n2_min'] = np.min(stats['u_n2'])
        return stats

    # Returns the accuracy on data = (X,Y). If data is None, returns the OOB-estimate
    def score(self, data=None):
        return 1.0-self.risk(data)
    def risk(self, data=None):
        if data is None and self._sample_mode is None:
            util.warn('Warning, MVBase.risk: No OOB data!')
            return 1.0
        if data is None:
            #### WARNING: not implemented, not used TODO.
            (preds,targs) = self._OOB
            return 1.0 #util.oob_estimate(self._rho, preds, targs)
        else:
            (X,Y) = data
            P = self.predict_all(X)
            return util.mv_risk(self._rho,P,Y)

    # Returns the Gibbs risk
    def gibbs_risk(self, data=None, incl_oob=True):
        rs, n = self.risks(data, incl_oob)
        return np.average(rs/n, weights=self._rho), np.min(n)
    def risks(self, data=None, incl_oob=True):
        incl_oob = incl_oob and self._sample_mode is not None
        if data is None and not incl_oob:
            util.warn('Warning, MVBase.risks: Missing data!')
            return None
        m     = len(self._estimators)
        n     = np.zeros((m,))
        risks = np.zeros((m,))
        
        if incl_oob:
            (preds,targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            orisk, on = util.oob_risks(preds, targs)
            n     += on
            risks += orisk

        if data is not None:
            assert(len(data)==2)
            X,Y = data
            P = self.predict_all(X)

            n             += X.shape[0]
            risks         += util.risks(P, Y)
        
        return risks, n

    # Returns the tandem risk
    def tandem_risk(self, data=None, incl_oob=True):
        trsk, n2 = self.tandem_risks(data, incl_oob)
        return np.average(np.average(trsk/n2, weights=self._rho, axis=1), weights=self._rho), np.min(n2)
    def tandem_risks(self, data=None, incl_oob=True):
        incl_oob = incl_oob and self._sample_mode is not None
        if data is None and not incl_oob:
            util.warn('Warning, MVBase.tandem_risks: Missing data!')
            return None
        m     = len(self._estimators)
        n2    = np.zeros((m,m))
        tandem_risks = np.zeros((m,m))
        
        if incl_oob:
            (preds,targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            otand, on2 = util.oob_tandem_risks(preds,targs)
            n2           += on2
            tandem_risks += otand

        if data is not None:
            assert(len(data)==2)
            X,Y = data
            P = self.predict_all(X)
            
            n2            += X.shape[0]
            tandem_risks  += util.tandem_risks(P, Y)

        return tandem_risks, n2 

    # Returns the disagreement
    def disagreement(self, data=None, incl_oob=True):
        dis, n2 = self.disagreements(data, incl_oob)
        return np.average(np.average(dis/n2, weights=self._rho, axis=1), weights=self._rho), np.min(n2)
    def disagreements(self, data=None, incl_oob=True):
        incl_oob = incl_oob and self._sample_mode is not None
        if data is None and not incl_oob:
            util.warn('Warning, MVBase.disagreements: Missing data!')
            return None
        m     = len(self._estimators)
        n2    = np.zeros((m,m))
        disagreements = np.zeros((m,m))
       
        if incl_oob:
            (preds,Y) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            odis, on2 = util.oob_disagreements(preds)
            n2            += on2
            disagreements += odis

        if data is not None:
            X = data
            P = self.predict_all(X)
            
            n2            += X.shape[0]
            disagreements += util.disagreements(P)

        return disagreements, n2 
