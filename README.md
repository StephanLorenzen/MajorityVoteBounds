# Majority Vote Classifiers With Performance Guarantees
This repository supplies a framework for implementing majority vote classifiers with performance guarantees. The implementation is used for experiments presented in [1,2,3,4]. When trained using bootstrapping or validation sets, theoretical guarantees based on PAC Bayesian theory will be computed, see [1,2,3,4,5].

The implementation is provided as a module, `mvb`, which provides a python class `MVBase`, which provides an interface for for implementing majority vote classifiers. `mvb` also provides three such implementations:

* RandomForestClassifier
* ExtraTreesClassifier
* SVMVotersClassifier
* MultiClassifierEnsemble

Each provide a majority vote classifier with an interface similar to `sklearn.ensemble.RandomForestClassifier` etc. The voters used in these implementations are based on various models from sklean: `sklearn.tree.DecisionTreeClassifier`, `sklearn.svm.SVC`, etc. [6]. 
Furthermore, the sub-module `mvb.data` can be used for reading data, while functions for computing bounds directly can be found in sub-module `mvb.bounds`.

Two directories with experiments are included in the repository:
* **NeurIPS2022** provides the experiments of [1].
* **NeurIPS2021** provides the experiments of [2].
* **NeurIPS2020** provides the experiments of [3].

Each directory contains a README with a description of how to run the experiments of the given paper, including downloading of data from various sources [7,8,9].

## Basic usage
Below follow a simple usage example of the `mvb` library:

```python
from mvb import RandomForestClassifier as RF
from mvb import data as mldata

X, Y = mldata.load('Letter:OQ')

rf = RF(n_estimators=100)
_ = rf.fit(X, Y)
bounds = rf.bounds()
```

## Acknowledgements
Some of the implementation in `mvb.bounds` is based on the implementation from [4].

## References
\[1\] [Wu and Seldin: Split-kl and PAC-Bayes-split-kl Inequalities for Ternary Random Variables (NeurIPS 2022)](https://arxiv.org/abs/2206.00706)

\[2\] [Wu, Masegosa, Lorenzen, Igel and Seldin: Chebyshev-Cantelli PAC-Bayes-Bennett Inequality for the Weighted Majority Vote (NeurIPS 2021)](https://arxiv.org/abs/2106.13624)

\[3\] [Masegosa, Lorenzen, Igel and Seldin: Second Order PAC-Bayesian Bounds for the Weighted Majority Vote (NeurIPS 2020)](https://arxiv.org/abs/2007.13532)

\[4\] [Lorenzen, Igel and Seldin: On PAC-Bayesian Bounds for Random Forests (ECML 2019)](https://arxiv.org/abs/1810.09746)

\[5\] Germain, Lacasse, Laviolette, Marchand and Roy: Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (JMLR 2015)

\[6\] [The sklearn.ensemble module](https://scikit-learn.org/stable/modules/ensemble.html)

\[7\] [The UCI Repository](https://archive.ics.uci.edu/ml/index.php)

\[8\] [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

\[9\] [Zalando Research](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
