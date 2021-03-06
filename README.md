# TDT4173 - Machine Learning Project
[![Documentation Status](https://readthedocs.org/projects/tdt4173/badge/?version=latest)](https://tdt4173.readthedocs.io/en/latest/?badge=latest)


Source code for the Machine Learning subject TDT4173. This repo contains
everything required to reproduce project results.
Supplementary website is deployed at the following [site](https://alexaor.github.io/tdt4173/).



## Project wizard
When all required dependencies are installed, the project can be run using a project wizard by running the `main.py`
file and then following the instructions showing up in the terminal. If desired its possible to execute a more 
raw implementation in the file `playground.py` and is intended as more of a doodling ground. Changing the hyperparameters
for the different methods can be done in the `gin` files in the config folder, one for [50]( https://github.com/alexaor/tdt4173/blob/main/configs/hyperparameters_50.gin) features and one for [all](https://github.com/alexaor/tdt4173/blob/main/configs/hyperparameters_all.gin) features.  
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

* python = "^3.8"
* numpy = "^1.18.5"
* scikit-learn = "^0.23.2"
* matplotlib = "^3.3.2"
* pandas = "^1.1.3"
* keras = "^2.4.3"
* tensorflow = "^2.3.1"
* gin-config = "^0.3.0"
* colorama = "^0.4.4"
* tensorflow-addons = "^0.11.2"
