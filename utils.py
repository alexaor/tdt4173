import pathlib
from os.path import join

from pandas import read_csv
from evaluation.evaluate import (
    plot_precision_recall,
    plot_roc_auc,
    plot_evaluation_result,
    plot_comparison,
    print_evaluation
)
from methods.models import Models
from preprocessing.preprocessor import create_data_set
from numpy import log

from configs.project_settings import DATASET_PATH


def get_dataset(filename):
    dataset = read_csv(join(DATASET_PATH, filename))
    x_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, -1].values
    return x_train, y_train


def get_models(input_shape, initial_bias, keys):
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


def fit_all_models(models, x_train, y_train):
    for model in models.keys():
        models[model].fit(x_train, y_train)
    return models


def get_all_predictions(models, x_test):
    models_proba = {}
    models_bool = {}
    for model in models.keys():
        if model != 'DNN':
            models_proba[model], models_bool[model] = models[model].predict(x_test)
    return models_proba, models_bool


def plot_training_curves(models, x_train, y_train, plotname, compare_criterion=False):
    print('Plotting learning curves for the methods ...')
    for model in models.keys():
        if model == 'Decision Tree':
            models[model].plot_learning_curves(x_train, y_train, plotname, compare_criterion)
        elif model != 'DNN':
            models[model].plot_learning_curves(x_train, y_train, plotname, False)
        else:
            models[model].plot_accuracy(plotname)
