#
# Implements ExtraTreesClassifier in the framework
#
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils import check_random_state

from . import util, mvbase

class ExtraTreesClassifier(mvbase.MVBounds):
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
            random_state=None
            ):
        self._max_depth        = max_depth
        self._actual_max_depth = max_depth

        prng = check_random_state(random_state)
        estimators = []
        for i in range(n_estimators):
            tree = Tree(criterion=criterion,
                        splitter="random",
                        max_features=max_features,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_depth=max_depth,
                        random_state=prng)
            estimators.append(tree)

        super().__init__(estimators, rho, sample_mode='bootstrap' if bootstrap else None, random_state=prng)

    def fit(self, X, Y):
        estimate = super().fit(X,Y)
        self._actual_max_depth = max([t.tree_.max_depth for t in self._estimators])
        return estimate

    def get_max_depth(self):
        return self._actual_max_depth
