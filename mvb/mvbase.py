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

#from mvb.bounds import muBernstein
from . import util
from .bounds import SH, PBkl, optimizeLamb, C1, C2, CTD, TND, optimizeTND, DIS, optimizeDIS, MU, optimizeMU, MUBernstein, optimizeMUBernstein
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

    # Remove classifiers with lowest weight (ties broken abitrarily)
    # if n != None: remove n classifiers with lowest weight
    # if b != None: remove classifiers with weight <= b
    # if use_oob, removes based on accuracy on OOB sets instead of weight 
    # Default: remove classifier with lowest weight
    # Guarantess at least one voter remains
    # Returns number of voters removed and score of weakest voter left
    def sparsify(self, n=None, b=None, use_oob=False):
        ws = self._rho
        if use_oob:
            ws,ns = self.risks()
            ws /= ns
            ws = 1.0 - ws
        ws = list(zip(list(range(len(ws))),ws))
        ws.sort(key=lambda x: -x[1])
        
        # Check params
        if n is None:
            if b is None:
                n,b = 1,-1.0
            else:
                n = len(ws)-1
        else:
            n = min(n,len(ws)-1)
            b = -1.0

        nrho, nest, noobs = [], [], []
        oob_preds, Y = self._OOB
        mins = 0.0
        for (i,s) in ws[:-n]:
            if s > b:
                nrho.append(self._rho[i])
                nest.append(self._estimators[i])
                noobs.append(oob_preds[i])
                mins = s
            else:
                break

        res = len(self._rho)-len(nrho)
        
        # Update and normalize rho
        self._rho = np.array(nrho)/sum(nrho)
        self._estimators = nest
        self._OOB = (noobs, Y)

        return res, mins


    # Optimizes the weights.
    def optimize_rho(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, options=None):
        if bound not in {"Lambda", "TND", "DIS", "MU", "MUBernstein"}:
            util.warn('Warning, optimize_rho: unknown bound!')
            return None
        if labeled_data is None and not incl_oob:
            util.warn('Warning, stats: Missing data!')
            return None

        if bound=='Lambda':
            risks, ns = self.risks(labeled_data, incl_oob)
            (bound, rho, lam) = optimizeLamb(risks/ns, np.min(ns))
            self._rho = rho
            return (bound,rho,lam)
        elif bound=='TND':
            tand, n2s = self.tandem_risks(labeled_data, incl_oob)
            (bound,rho,lam) = optimizeTND(tand/n2s, np.min(n2s), options=options)
            self._rho = rho
            return (bound, rho, lam)
        elif bound=='DIS':
            ulX = unlabeled_data
            if labeled_data is not None:
                ulX = labeled_data[0] if ulX is None else np.concatenate((ulX,labeled_data[0]), axis=0)
            risks, ns = self.risks(labeled_data, incl_oob)
            dis, n2s = self.disagreements(ulX, incl_oob)
            (bound,rho,lam,gam) = optimizeDIS(risks/ns,dis/n2s,np.min(ns),np.min(n2s),options=options)
            self._rho = rho
            return (bound, rho, lam, gam)
        elif bound == 'MU':
            risks, ns = self.risks(labeled_data, incl_oob)
            tand, n2s = self.tandem_risks(labeled_data, incl_oob)
            (bound,rho,mu,lam,gam) = optimizeMU(tand/n2s,risks/ns,np.min(ns),np.min(n2s),options=options)
            self._rho = rho
            return (bound, rho, mu, lam, gam)
        else:
            (bound,rho,mu,lam,gam) = optimizeMUBernstein(self, labeled_data, incl_oob,options=options)
            self._rho = rho
            return (bound, rho, mu, lam, gam)


    # Computes the given bound ('SH', 'PBkl', 'C1', 'C2', 'CTD', 'TND', 'DIS', 'MU').
    # A stats object or the relevant data must be given as input (unless classifier trained
    # with bagging, in which case this data can be used).
    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, stats=None):
        if bound not in ['SH', 'PBkl', 'C1', 'C2', 'CTD', 'TND', 'DIS', 'MU','MUBernstein']:
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
            #elif bound=='MU':
            #    return MU(stats['tandem_risk'], stats['gibbs_risk'], stats['n_min'], stats['n2_min'], KL, stats['mu_mub'])
            elif bound == 'MUBernstein':
                return MUBernstein(self, labeled_data, incl_oob, KL, stats['mu_bern'])
            #elif bound == 'MUVarBernstein':
            #    mutandem_risk, vartandem_risk, n2 = self.mutandem_risk(stats['mu'], labeled_data, incl_oob)
            #    varMUBound, _ = muBernstein.varMUBernstein(vartandem_risk, n2, KL, stats['mu'])
            #    return varMUBound
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
                return CTD(grisk, tand, n_min, n2_min, KL)
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
            elif bound=='MU':
                # Store full OOB
                full_OOB = self._OOB
                # Estimate MU and reduce OOB
                mu, self._OOB = self.estimate_mu()
                # Estimate randomized risk and tandem risk (with reduced OOB)
                grisk, n_min = self.gibbs_risk(labeled_data, incl_oob)
                tand, n2_min = self.tandem_risk(labeled_data, incl_oob)
                # Reset OOB
                self._OOB = full_OOB
                return MU(tand, grisk, n_min, n2_min, KL, mu_grid=[mu])
            elif bound == 'MUBernstein':
                mu = 0.0
                mutandem_risk, vartandem_risk, n2 = self.mutandem_risk(mu, labeled_data, incl_oob)
                return MUBernstein(mutandem_risk, vartandem_risk, n2, KL, mu)
            elif bound == 'MUVarBernstein':
                mu = 0.0
                mutandem_risk, vartandem_risk, n2 = self.mutandem_risk(mu, labeled_data, incl_oob)
                varMUBound, _ = muBernstein.varMUBernstein(vartandem_risk, n2, KL, mu)
                return varMUBound

            else:
                return None
    
    # Compute all bounds, given relevant stats object or data
    def bounds(self, labeled_data=None, unlabeled_data=None, incl_oob=True, stats=None):
        results = dict()
        options = dict()
        if stats is None:
            stats = self.stats(labeled_data, unlabeled_data, incl_oob)

        results['PBkl'] = self.bound('PBkl', stats=stats)
        (results['TND'], options['TandemUB'])  = self.bound('TND', stats=stats)
        results['CTD']  = self.bound('CTD', stats=stats)
        if incl_oob:
            #(results['MU'], options['mu_mub'], options['muTandemUB']) = self.bound('MU', stats=stats)
            (results['MUBernstein'], options['mu_bern'], options['mutandem_risk'], options['vartandem_risk'], options['varUB'], options['bernTandemUB']) = self.bound('MUBernstein', stats=stats)
        if labeled_data is not None or (stats is not None and 'mv_risk' in stats):
            results['SH'] = self.bound('SH', stats=stats)
        if self._classes.shape[0]==2:
            results['C1']  = self.bound('C1', stats=stats)
            results['C2']  = self.bound('C2', stats=stats)
            results['DIS'] = self.bound('DIS', stats=stats)
        
        stats = self.aggregate_stats(stats, options)
        return results, stats
    
    # Compute stats object
    def stats(self, labeled_data=None, unlabeled_data=None, incl_oob=True, options=None):
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
        
            
        """
        if incl_oob:
            # Estimate MU and risks and tandem risks with [r]educed OOB
            full_OOB = self._OOB
            stats['mu'], self._OOB = self.estimate_mu()
            stats['r_risks'], stats['r_n'] = self.risks(labeled_data, incl_oob)
            stats['r_risks'] /= stats['r_n']
            
            stats['r_tandem_risks'], stats['r_n2'] = self.tandem_risks(labeled_data, incl_oob)
            stats['r_tandem_risks'] /= stats['r_n2']
            # Reset OOB
            self._OOB = full_OOB

        else:
            # Set to non-reduced stats => should just give the tandem bound.
            stats['mu'] = 0.0
            stats['r_risks'], stats['r_n'] = stats['risks'], stats['n']
            stats['r_tandem_risks'], stats['r_n2'] = stats['tandem_risks'], stats['n2']
        """

        return self.aggregate_stats(stats, options)
    
    # (Re-)Aggregate stats object. Useful if weighting has changed.
    def aggregate_stats(self, stats, options=None):
        if 'mv_preds' in stats:
            stats['mv_risk'] = util.mv_risk(self._rho, stats['mv_preds'][0], stats['mv_preds'][1]) 
        
        stats['gibbs_risk'] = np.average(stats['risks'], weights=self._rho)
        stats['n_min'] = np.min(stats['n'])

        stats['tandem_risk'] = np.average(np.average(stats['tandem_risks'], weights=self._rho, axis=0), weights=self._rho)
        stats['disagreement'] = np.average(np.average(stats['disagreements'], weights=self._rho, axis=0), weights=self._rho)
        stats['n2_min'] = np.min(stats['n2'])
        
        pi = util.uniform_distribution(len(self._estimators))
        stats['KL'] = util.kl(self._rho, pi)
       
        # Unlabeled
        stats['u_disagreement'] = np.average(np.average(stats['u_disagreements'], weights=self._rho, axis=0), weights=self._rho)
        stats['u_n2_min'] = np.min(stats['u_n2'])
        
        # for 'MUBernstein' bounds
        options = dict() if options is None else options
        for key in options:
            stats[key] = options[key]
        stats['mu_bern'] = options.get('mu_bern', (0.0,))

        """
        # Reduced OOB
        stats['r_gibbs_risk'] = np.average(stats['r_risks'], weights=self._rho)
        stats['r_n_min'] = np.min(stats['r_n'])
        stats['r_tandem_risk'] = np.average(np.average(stats['r_tandem_risks'], weights=self._rho, axis=0), weights=self._rho)
        stats['r_n2_min'] = np.min(stats['r_n2'])
        """
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

    # Returns the Gibbs risk and n_min
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

    def mutandem_risk(self, mu, data=None, incl_oob=True):
        mutrisks, musquaretandem_risks, n2 = self.mutandem_risks(mu, data, incl_oob)
        mutandemrisk = np.average(np.average(mutrisks / n2, weights=self._rho, axis=1), weights=self._rho)

        vartandemrisks = (n2/(n2-1))*(musquaretandem_risks / n2 - np.square(mutrisks / n2))
        vartandemrisk = np.average(np.average(vartandemrisks, weights=self._rho, axis=1), weights=self._rho)

        return mutandemrisk, vartandemrisk, np.min(n2)

    def mutandem_risks(self, mu, data=None, incl_oob=True):
        incl_oob = incl_oob and self._sample_mode is not None
        if data is None and not incl_oob:
            util.warn('Warning, MVBase.mutandem_risks: Missing data!')
            return None
        m = len(self._estimators)
        n2 = np.zeros((m, m))
        mutandem_risks = np.zeros((m, m))
        musquaretandem_risks = np.zeros((m, m))

        if incl_oob:
            (preds, targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            omutand, omusquaretand, on2 = util.oob_mutandem_risks(preds, targs, mu)
            n2 += on2
            mutandem_risks += omutand
            musquaretandem_risks += omusquaretand

        if data is not None:
            assert (len(data) == 2)
            X, Y = data
            P = self.predict_all(X)

            n2 += X.shape[0]
            mutand, musquaretand = util.mutandem_risks(P, Y, mu)
            mutandem_risks += mutand
            musquaretandem_risks += musquaretand

        return mutandem_risks, musquaretandem_risks, n2

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
    
    """
    # Estimates mu using a single sample from each oob set
    # Returns: estimated mu ~= E_u[L(h)] and reduced oob set
    def estimate_mu(self):
        if self._OOB is None:
            util.warn('Warning, MVBase.estimate_mu: Missing data!')
            return 0.0, None
        (OOBs, Y)   = self._OOB
        new_OOBs    = []
        mu_estimate = 0
        # Not vectorized, but should be fast enough
        for (mask, preds) in OOBs:
            # Do not make changes to originals
            mask, preds = mask.copy(), preds.copy()
            
            # Sample a single data point from this oob set
            # Indexes of non-zero
            idxs = np.flatnonzero(mask)
            # Sample one index at random
            idx = np.random.choice(idxs)
            # Add l(preds[idx],Y[idx]) to mu_estimate
            mu_estimate += 1 if preds[idx]!=Y[idx] else 0
            # Set mask and oob_pred to zero
            mask[idx], preds[idx] = 0, 0
            new_OOBs.append((mask,preds))
            
        return mu_estimate/len(OOBs), (new_OOBs, Y)
        """
