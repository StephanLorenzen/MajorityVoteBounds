#
# Implements Artificial RandomForestClassifier where trees make independent errors
#
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils import check_random_state
import math
from . import util, mvbase

class ArtificialRandomForestClassifier(mvbase.MVBounds):
    def __init__(
            self,
            gibbs_error,
            n_estimators,
            dataset_size,
            num_classes,
            random_state = None
            ):

        self._rho = util.uniform_distribution(n_estimators)

        self._estimators = [None] * n_estimators

        self._sample_mode = 'boostrap'

        self._prng = check_random_state(random_state)

        # We randomly simluate the class labels.
        Y = self._prng.choice(range(num_classes),size = dataset_size, replace=True)

        preds = [None]*n_estimators
        for H in range(n_estimators):
            # samplesH will contain which samples are in the bag and ones are out of the bag for the hipothesis H.
            samplesH = np.ones(dataset_size)

            # predsH will contain the predictions of the hipothesis H for the oob samples.
            predsH = np.zeros(dataset_size)

            # We obtain the bootstrap sample for the hipothesis H.
            bootstrap_sample = self._prng.choice(range(dataset_size), size=dataset_size, replace=True)

            # We annotate the bootstrap samples with a 0.
            samplesH[bootstrap_sample] = 0

            # We generate the error of the hipothesis H according the given gibbs_error.
            herrors = self._prng.choice(2, size=dataset_size, replace=True, p=[1 - gibbs_error, gibbs_error]) == 1

            # We generate a boolean vector indicating where are the out-of-the-bag samples
            oob_samples = (samplesH == 1)

            # We generate a boolean vector indicating where are the errors of the hipothesis H in the out-of-the-bag samples
            oob_errors = np.logical_and(oob_samples, herrors)

            # We generate a error pattern
            errorsY = np.remainder(Y + self._prng.randint(num_classes - 1, size=dataset_size) + 1, num_classes)

            # The predictions for those not oob samples are set to 0.
            predsH[np.logical_not(oob_samples)] = 0

            # We set the wrong predictions for the oob samples which were missclassified.
            predsH[oob_errors] = errorsY[oob_errors]

            # We set the right predictions for the oob samples which were no missclassified.
            predsH[np.logical_and(oob_samples, np.logical_not(oob_errors))] = Y[np.logical_and(oob_samples, np.logical_not(oob_errors))]

            # We store the oob indicator and the predictions
            preds[H] = (samplesH, predsH)

        self._OOB = (preds,Y)

    def fit(self, X, Y):
        return None

    def predict(self, X, Y=None):
        return None, None
