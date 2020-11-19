import pathlib
from os.path import join

from pandas import read_csv
from gin import parse_config_file

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


def main():
    keys = ['Random Forest', 'DNN', 'Ada Boost', 'Decision Tree']
    dnn_confusion_matrix = None
    x_train, y_train = get_dataset('02_features_50_train.csv')
    x_test, y_test = get_dataset('02_features_50_test.csv')

    # Calculate initial bias for DNN
    pos = sum(y_train) + sum(y_test)
    neg = (len(y_train) + len(y_test)) - pos
    initial_bias = log([pos/neg])

    models = get_models((len(x_train[0]),), initial_bias, keys)
    models = fit_all_models(models, x_train, y_train)

    # === Loading models ===
    # models['DNN'].load_model('dnn_features_all.h5')
    # models['Ada Boost'].load_model('ab_tuned_features_all.sav')
    # models['Decision Tree'].load_model('dt_tuned_features_all.sav')
    # models['Random Forest'].load_model('rf_tuned_features_all.sav')

    # === Make predictions ===
    models_proba, models_bool = get_all_predictions(models, x_test)
    if 'DNN' in keys:
        models_proba['DNN'], dnn_confusion_matrix = models['DNN'].evaluate(x_test, y_test)
        models_bool['DNN'] = models_proba['DNN'].copy()

    print('\n\n================ Evaluation ================\n')
    # === Plot training evaluations of DNN ===
    # models['DNN'].plot_cross_evaluation(5, x_train, y_train, x_test, y_test, 'all_features.png')
    # models['DNN'].plot_model('network_50.png')
    # models['DNN'].plot_training_evaluation('training_evaluation_50.png')

    # === Plot training evaluations of the SKLearn methods ===
    # plot_training_curves(models, x_train, y_train, 'all_features.png', True)

    # === Plot and print differnt training evaluations ===
    # print_evaluation(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix, filename='features_50.txt')
    # plot_roc_auc(y_test, models_proba, 'roc_features_50.png')
    # plot_evaluation_result(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix)
    plot_comparison(y_test, models_bool, ['Cohen kappa', 'f1', 'Precision', 'Recall', 'Number of yes'],
                    dnn_conf_matrix=dnn_confusion_matrix, filename='comparison_features_50_no_class_weight.png')
    # plot_precision_recall(y_test, models_proba, 'pr_features_50.png')

    # === Saving models ===
    # models['Decision Tree'].save_model('dt_tuned_50_features.sav')
    # models['Ada Boost'].save_model('ab_tuned_all_features.sav')
    # models['Random Forest'].save_model('rf_tuned_50_features.sav')
    # models['DNN'].save_model('dnn_features_50.h5')


if __name__ == "__main__":
    # create_data_set('features_50', n_features=50)
    parse_config_file('configs/hyperparameters_50.gin')
    main()
