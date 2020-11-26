from os.path import join
from pandas import read_csv
from methods.models import Models
from configs.project_settings import DATASET_PATH
from preprocessing.preprocessor import create_data_set

from typing import Tuple
from numpy import ndarray


def get_dataset(filename) -> Tuple[ndarray, ndarray]:
    """
    Splits the output from the input in the datasets, and return the two resulting arrays.

    Parameters
    ----------
    filename : string
        Name of the file to be split

    Returns
    -------
    x_train : numpy.ndarray
        A numpy.ndarray matrix consisting of n_samples and n_features, used
        as training input samples
    y_train : array
        An array of output sample values used during training
    """

    dataset = read_csv(join(DATASET_PATH, filename))
    x_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, -1].values
    return x_train, y_train


def get_models(input_shape, initial_bias, keys) -> dict:
    """
    Creates a object of all classifiers stated in "keys" and returns them in a dictionary.

    Parameters
    ----------
    input_shape : tuple of ints
        Shape of the samples, I.e., the number of features in the samples
    initial_bias : list of floats
        A list with only one float that sets the initial bias on the output layer
    keys : list of strings
        A list of names of models. NOTE: its case sensitive

    Returns
    -------
    models : dictionary of classifiers
        A dictionary of object classifiers, sated in keys.
    """

    models = {}
    if 'Random Forest' in keys:
        models['Random Forest'] = Models.random_forest()
    if 'Ada Boost' in keys:
        models['Ada Boost'] = Models.ada_boost()
    if 'Decision Tree' in keys:
        models['Decision Tree'] = Models.decision_tree()
    if 'DNN' in keys:
        models['DNN'] = Models.dnn(input_shape=input_shape, initial_bias=initial_bias)
    return models


def fit_all_models(models, x_train, y_train) -> dict:
    """
    Fits all classifiers in models with the dataset.

    Parameters
    ----------
    models : dictionary of classifiers
        A dictionary of object classifiers
    x_train : numpy.ndarray
        A numpy.ndarray matrix consisting of n_samples and n_features, used
        as training input samples
    y_train : array
        An array of output sample values used during training

    Returns
    -------
    models : dictionary of classifiers
        A dictionary of object classifiers, where all the classifiers are fitted
    """

    for model in models.keys():
        models[model].fit(x_train, y_train)
    return models


def get_all_predictions(models, x_test) -> Tuple[dict, dict]:
    """
    Creates the predictions the the classifiers in models with the given data.

    Parameters
    ----------
    models : dictionary of classifiers
        A dictionary of object classifiers, sated in keys.
    x_test : numpy.ndarray
        A numpy.ndarray matrix consisting of n_samples and n_features, used
        as training input samples

    Returns
    -------
    models_proba : dict of list of floats
        A dictionary with the classifier name as key and the predicted probabilities as values
    models_bool : dict of list of bools
        A dictionary with the classifier name as key and the predicted bool as values
    """

    models_proba = {}
    models_bool = {}
    for model in models.keys():
        if model != 'DNN':
            models_proba[model], models_bool[model] = models[model].predict(x_test)
    return models_proba, models_bool


def plot_training_curves(models, x_train, y_train, plotname, compare_criterion=False) -> None:
    """
    Creates k-fold cross evaluation plots for the methods in models.

    Parameters
    ----------
    models : dictionary of classifiers
        A dictionary of object classifiers, sated in keys.
    x_train : numpy.ndarray
        A numpy.ndarray matrix consisting of n_samples and n_features, used
        as training input samples
    y_train : array
        An array of output sample values used during training
    plotname : string
        Name of the plot, required to have the file extension `.png`
    compare_criterion :  bool
        If the learning curve should compare the two different criterion or not
    """

    print('Plotting learning curves for the methods ...')
    for model in models.keys():
        if model == 'Decision Tree':
            models[model].plot_learning_curves(x_train, y_train, plotname, compare_criterion)
        elif model != 'DNN':
            models[model].plot_learning_curves(x_train, y_train, plotname, False)


def setup_data(feature_list=(50,)) -> None:
    """
    This module creates datasets for feature-selection with 50 features in addition
    to a set with all features included. Output format is 'features_n' where n is
    the number of features selected. Dataset with no features selection includes the
    name all_features

    Parameters
    ----------
    feature_list : tuple
        How many features the dataset should have
    """

    for feature_number in feature_list:
        print(f"Creating datasets with {feature_number} features")
        create_data_set(f"features_{feature_number}", n_features=feature_number)
        print("Datasets created")
    print("Creating datasets with no feature selections")
    create_data_set(f"all_features")
    print("Datasets created")
