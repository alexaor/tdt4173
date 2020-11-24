import sys
from gin import parse_config_file
from numpy import log
from evaluation.evaluate import (
    plot_precision_recall,
    plot_roc_auc,
    plot_evaluation_result,
    plot_comparison,
)
from utils import (
    get_dataset,
    get_models,
    get_all_predictions,
    fit_all_models,
    plot_training_curves,
    setup_data,
)


def kill(msg, callback=None):
    print(f"ERROR: {msg}")
    if callback is None:
        sys.exit(1)
    else:
        callback()


def on_interrupt(func, callback):
    try:
        func()
    except KeyboardInterrupt:
        callback()


def main(name_of_dataset):
    keys = ['Random Forest', 'DNN', 'Ada Boost', 'Decision Tree']
    dnn_confusion_matrix = None
    x_train, y_train = get_dataset(f'02_{name_of_dataset}_train.csv')
    x_test, y_test = get_dataset(f'02_{name_of_dataset}_test.csv')

    # Calculate initial bias for DNN
    pos = sum(y_train) + sum(y_test)
    neg = (len(y_train) + len(y_test)) - pos
    initial_bias = log([pos / neg])

    models = get_models((len(x_train[0]),), initial_bias, keys)
    models = fit_all_models(models, x_train, y_train)

    # === Make predictions ===
    models_proba, models_bool = get_all_predictions(models, x_test)
    if 'DNN' in keys:
        models_proba['DNN'], dnn_confusion_matrix = models['DNN'].evaluate(x_test, y_test)
        models_bool['DNN'] = models_proba['DNN'].copy()

    # === Plot training evaluations of DNN ===
    models['DNN'].plot_cross_evaluation(5, x_train, y_train, x_test, y_test, f'{name_of_dataset}.png')
    models['DNN'].plot_training_evaluation(f'training_evaluation_{name_of_dataset}.png')

    # === Plot training evaluations of the SKLearn methods ===
    plot_training_curves(models, x_train, y_train, f'{name_of_dataset}.png', True)

    # === Plot different training evaluations ===
    plot_roc_auc(y_test, models_proba, f'roc_{name_of_dataset}.png')
    plot_evaluation_result(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix)
    plot_comparison(y_test, models_bool, ['Cohen kappa', 'f1', 'Precision', 'Recall', 'Number of yes'],
                    dnn_conf_matrix=dnn_confusion_matrix, filename=f'comparison_{name_of_dataset}_no_class_weight.png')
    plot_precision_recall(y_test, models_proba, f'pr_{name_of_dataset}.png')


def menu():
    print()
    print("Options:")
    print("  [1]: Create datasets")
    print("  [2]: Run with 50 features")
    print("  [3]: Run with all features")
    print("  [q]: Quit")

    pindex = input("> ")

    if pindex == "q":
        print("bye")
        sys.exit(0)

    try:
        pindex = int(pindex)
    except ValueError:
        kill(f"'{pindex}' is not a valid option", menu)

    if pindex == 1:
        on_interrupt(setup_data, menu)
    elif pindex == 2:
        parse_config_file('configs/hyperparameters_50.gin')
        on_interrupt(main("features_50"), menu)
    elif pindex == 3:
        parse_config_file("configs/hyperparameters_all.gin")
        on_interrupt(main("all_features"), menu)
    else:
        print("Invalid option")
    menu()


if __name__ == "__main__":
    print("Welcome to this TDT4173 project's commandline tool")
    print("you will be presented with a commandline wizard to quickly bootstrap the project")
    print("If this is your first time running you will need to generate the datasets as presented in option [1]")
    on_interrupt(menu, lambda: print("\nbye"))
