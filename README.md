# A Random Forest With Performance Guarantees
This repository supplies a wrapper implementation for random forests, and is used for experiments conducted in [1]. When trained using bootstrapping or validation sets, theoretical guarantees based on PAC Bayesian theory will be computed, see [1,2] for the theory.

The implementation is provided as a module, `rfb`, which provides a python class `RandomForestWithBounds` similar to `sklearn.ensemble.RandomForestClassifier` etc. The class wraps an underlying decision tree/random forest implementation. The module allows for selecting this underlying implementation from among `sklearn.ensemble.RandomForestClassifer`, `sklearn.ensemble.ExtraTreesClassifier` [3] and the `Woody` random forest implementation [4].
Furthermore, the submodule `rfb.data` can be used for reading data, while functions for computing bounds directly can be found in submodule `rfb.bounds`. Note, that some of the implementation in `rfb.bounds` are taken from the implementation for the paper [2].

In addition, three files are included in the repository:
* **getdata.sh** is a shell script used to download the data sets from the UCI repository [5] and place them in the directory `data`.
* **sample.py** contains a sample usage example: a 100-tree random forest is trained on the `Letter:AB` data set with and without using a validation set, and bounds are computed.
* **experiment.py** runs the main experiment of [1] and creates the main table in the directory `out`.

## Basic usage
Belows follow a simple usage example. See `sample.py` and `experiment.py` for more.

```python
import rfb

X, Y = rfb.data.load('Letter:AB')

rf = RFWB(n_estimators=100, lib='sklearn-rfc')
oob_estimate, bounds = rf.fit(X, Y)
```

Here, bounds is a dictionary with keys `PBkl`, `C1` and `C2`.

## References
\[1\] [Lorenzen, Igel and Seldin: On PAC-Bayesian Bounds for Random Forests (ECML 2019)](https://arxiv.org/abs/1810.09746)

\[2\] Germain, Lacasse, Laviolette, Marchand and Roy: Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (JMLR 2015)

\[3\] [The sklearn.ensemble module](https://scikit-learn.org/stable/modules/ensemble.html)

\[4\] Gieseke and Igel: Training big random forests with little resources (SIGKDD 2018)

\[5\] [The UCI Repository](https://archive.ics.uci.edu/ml/index.php)
