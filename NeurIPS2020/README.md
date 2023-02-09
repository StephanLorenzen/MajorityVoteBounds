# Experiments for: "Second Order PAC-Bayesian Bounds for the Weighted Majority Vote"
This directory contains the code for running the experiment of the paper [1]:

[Masegosa, Lorenzen, Igel and Seldin: Second Order PAC-Bayesian Bounds for the Weighted Majority Vote (NeurIPS 2020)](https://arxiv.org/abs/2007.13532)

## Usage
To run the experiments, run `make`. This will download the data [2,3,4] needed for the experiments.
Experiments can now be run by using the `uniform.py`, `optimize.py` and `unlabeled.py` python scripts in the folder. See the files for how-to. Output files will be created in directory `out/`.
The full experiments from [1] can be run as follows:

* Uniform weighted RF (full bagging): `make uniform`
* Uniform weighted RF (reduced bagging): `make uniform-reduced`
* Optimized weighted RF (full bagging): `make optimize`
* Optimized weighted RF (reduced bagging): `make optimize-reduced`
* RF with unlabeled data: `make unlabeled`

## References
\[1\] [Masegosa, Lorenzen, Igel and Seldin: Second Order PAC-Bayesian Bounds for the Weighted Majority Vote (NeurIPS 2020)](https://arxiv.org/abs/2007.13532)

\[2\] [The UCI Repository](https://archive.ics.uci.edu/ml/index.php)

\[3\] [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

\[4\] [Zalando Research](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
