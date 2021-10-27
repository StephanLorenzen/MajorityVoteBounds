# Experiments for: "Chebyshev-Cantelli PAC-Bayes-Bennett Inequality for the Weighted Majority Vote"
This directory contains the code for running the experiment of the paper [1]:

[Wu, Masegosa, Lorenzen, Igel and Seldin: Chebyshev-Cantelli PAC-Bayes-Bennett Inequality for the Weighted Majority Vote (NeurIPS 2021)](https://arxiv.org/abs/2106.13624)

## Usage
To run the experiments, run `make`. This will download the data [6,7,8] needed for the experiments.
Experiments can now be run by using the `optimize.py` python script in the folder. See the files for how-to. Output files will be created in directory `out/`.
The full experiments from [3] can be run as follows:

* Optimization of RF: `make optimize_rfc`
* Optimization of heterogeneous classifier ensemble: `make optimize_mce`
* Simulation (incl. generation of plot) of comparision to oracle bounds: `make artificial`

## Plots
After running the experiments, the plots and LaTeX tables of the paper can be generated by `cd`ing to the `plots` directory and running:

* Plots for optimized RF: `make optimize_rfc`
* Plots for optimized heterogeneous classifier ensemble: `make optimize_mce`

Output figures and tables will be generated in the subdirectories `figure` and `table`.