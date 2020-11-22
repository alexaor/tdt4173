# TDT4173 - Machine Learning Project
[![Documentation Status](https://readthedocs.org/projects/tdt4173/badge/?version=latest)](https://tdt4173.readthedocs.io/en/latest/?badge=latest)


Source code for the Machine Learning subject TDT4173. This repo contains
everything required to reproduce project results, in addition to supplementary
documentation which is hosted at [readthedocs.org](https://tdt4173.readthedocs.io/en/latest/?badge=latest).

## Dependencies
This project is heavily reliant on third-party libraries for implementations
of classifiers and configuration of said classifiers, a complete list of required
packages is included in this section

### Use of dependencies
scikit-learn and keras/tensorflow supply us with class implementations of 
different machine learning models we are implementing, in addition to
the `tensorflow-addons` module which is used to calculate Cohen's kappa on the predictions from tensorflow models. [`gin-config`](https://github.com/google/gin-config) is a lightweight 
library to allow for injection of hyperparameters through a config file which is
located in the `configs/` directory. `colorama` is used in order to colorize
terminal output in order to raise warnings.


### List over dependencies
Dependencies can be installed with your favorite virtualenvironment and package 
manager, there is also a `requirements.txt` for simple pip install with virtualenv
if desired.

* Python 3.8
* scikit-learn
* matplotlib
* keras
* gin-config
* colorama
* tensorflow-addons
