# Majority Vote Classifiers With Performance Guarantees
This repository supplies a framework for implementing majority vote classifiers with performance guarantees. The implementation is used for experiments presented in [1,2]. When trained using bootstrapping or validation sets, theoretical guarantees based on PAC Bayesian theory will be computed, see [1,2,3].

The implementation is provided as a module, `mvb`, which provides a python class `MVBase`, which provides an interface for for implementing majority vote classifiers. `mvb` also provides three such implementations:

* RandomForestClassifier
* ExtraTreesClassifier
* SVMVotersClassifier

Each provide a majority vote classifier with an interface similar to `sklearn.ensemble.RandomForestClassifier` etc. The voters used in these implementations are based on `sklearn.tree.DecisionTreeClassifier` and `sklearn.svm.SVC` [4]. 
Furthermore, the sub-module `mvb.data` can be used for reading data, while functions for computing bounds directly can be found in sub-module `mvb.bounds`. Note, that some of the implementation in `mvb.bounds` is taken from the implementation from [3].

In addition, two directories are included in the repository:
* **sample** provides a sample usage example: A 100-tree random forest is trained on the `Letter:OQ` data set with and without using a validation set, and bounds are computed. To run the sample, simply go to the directory and run `make` followed by `make run`.
* **experiments** provides the experiments of [1]. To run the experiments, go to the directory and run `make`. This will download the data [5,6,7] needed for the experiments.
Experiments can now be run by using the `uniform.py`, `optimize.py` and `unlabeled.py` python scripts in the folder. See the files for how-to. Output files will be created in directory `experiments/out/`.
The full experiments from [1] can be run as follows:
	* Uniform weighted RF (full bagging): `make uniform`
	* Uniform weighted RF (reduced bagging): `make uniform-reduced`
	* Optimized weighted RF (full bagging): `make optimize`
	* Optimized weighted RF (reduced bagging): `make optimize-reduced`
	* RF with unlabeled data: `make unlabeled`

	The directory also contains the sub-directory `NeurIPS_results`, which contains the numerical outputs reported in [1].

## Basic usage
Below follow a simple usage example. See `sample.py` for more.

```python
from mvb import RandomForestClassifier as RF
from mvb import data as mldata

X, Y = mldata.load('Letter:OQ')

rf = RF(n_estimators=100)
_ = rf.fit(X, Y)
bounds = rf.bounds()
```

Here, bounds is a dictionary with keys `PBkl`, `C1`, `C2`, `CTD` and `TND`.

## References
\[1\] [Masegosa, Lorenzen, Igel and Seldin: Second Order PAC-Bayesian Bounds for the Weighted Majority Vote (NeurIPS 2020)](https://arxiv.org/abs/2007.13532)

\[2\] [Lorenzen, Igel and Seldin: On PAC-Bayesian Bounds for Random Forests (ECML 2019)](https://arxiv.org/abs/1810.09746)

\[3\] Germain, Lacasse, Laviolette, Marchand and Roy: Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (JMLR 2015)

\[4\] [The sklearn.ensemble module](https://scikit-learn.org/stable/modules/ensemble.html)

\[5\] [The UCI Repository](https://archive.ics.uci.edu/ml/index.php)

\[6\] [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

\[7\] [Zalando Research](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
