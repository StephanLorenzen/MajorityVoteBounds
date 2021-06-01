#
# Implements Multi-Classifier Ensemble in the framework
#
import numpy as np
from . import util, mvbase
from sklearn.utils import check_random_state


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# List of classification models
classifiers = [
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5),
    KNeighborsClassifier(n_neighbors=5, weights='distance'),
    DecisionTreeClassifier(),
    LogisticRegression(),
    GaussianNB()
]

class MultiClassifierEnsemble(mvbase.MVBounds):
    def __init__(
            self,
            n_estimators,
            rho=None,
            sample_mode="bootstrap",
            random_state=None
            ):
        self._actual_n_estimators = len(classifiers)

        prng = check_random_state(random_state)

        estimators = []
        for i in range(len(classifiers)):
            model = classifiers[i]
            estimators.append(model)

        super().__init__(estimators, rho=rho, sample_mode=sample_mode, random_state=prng)

    # return the number of estimators
    def get_n_estimators(self):
        return self._actual_n_estimators
    
