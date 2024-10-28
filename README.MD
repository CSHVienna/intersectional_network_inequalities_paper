# Code for reproducing the results and figures from the paper *Intersectional inequalities in social networks*

by
S. Martin-Gutierrez,
M. N. Cartier van Dissel,
and
F. Karimi

## Abstract

Social networks are shaped by complex, intersecting identities that drive our connection preferences. These preferences weave networks where certain groups hold privileged positions, while others become marginalized. While previous research has examined the impact of single-dimensional identities on inequalities of social capital, social disparities accumulate nonlinearly, further harming individuals at the intersection of multiple disadvantaged groups. However, how multidimensional connection preferences affect network dynamics and in what forms they amplify or attenuate inequalities remains unclear.
In this work, we systematically analyze the impact of multidimensionality on social capital inequalities through the lens of intersectionality. To this end, we operationalize several notions of intersectional inequality in networks. Using a network model, we reveal how attribute correlation (or consolidation) combined with biased multidimensional preferences lead to the emergence of counterintuitive patterns of inequality that were unobserved in one-dimensional systems. We calibrate the model with real-world high school friendship data and derive analytical closed-form expressions for the predicted inequalities, finding that the model's predictions match the observed data with remarkable accuracy. These findings hold significant implications for addressing social disparities and inform strategies for creating more equitable networks. 

## Paper code

The code is divided betweeen the [`multisoc`](https://github.com/CSHVienna/multisoc) python package (we include the exact version used in the paper in a folder, no need to install the package), some auxiliary python scripts to perform the paper computations and helper functions for some of the figures, and [Jupyter notebooks](http://jupyter.org/) to generate the figures.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/CSHVienna/intersectional_network_inequalities_paper

or [download a zip file](https://github.com/CSHVienna/intersectional_network_inequalities_paper/archive/master.zip).

## Dependencies

The easiest way to setup the prerequisites to reproduce the code is to install an [Anaconda Python](https://www.anaconda.com/download) distribution and create a new Python environment with the required modules by running the following commands in the Anaconda terminal:

```
conda create --name intersectional --file requirements.txt
```

And **activate the environment** before running the paper code:

```
conda activate intersectional
``` 

## Reproducing the results

1. Perform the paper simulations by running:

```
python perform_paper_simulations.py
```

This is necessary to reproduce some of the paper results, but precomputed simulation results are available in the folder `paper_results`.

2. To generate the figures, open `jupyter lab` (e.g. by running the command `jupyter lab` in the Anaconda terminal in the folder containing the paper code). The notebook is divided into cells (some have text while others have code). Each cell can be executed using `Shift + Enter`. Executing text cells does nothing and executing code cells runs the code
and produces its outputs. To execute the whole notebook, run all cells in order.
