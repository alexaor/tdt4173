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
)


def main():
    keys = ['Random Forest', 'DNN', 'Ada Boost', 'Decision Tree']
    dnn_confusion_matrix = None
    x_train, y_train = get_dataset('02_features_50_train.csv')
    x_test, y_test = get_dataset('02_features_50_test.csv')

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
    models['DNN'].plot_cross_evaluation(5, x_train, y_train, x_test, y_test, 'all_features.png')
    models['DNN'].plot_training_evaluation('training_evaluation_50.png')

    # === Plot training evaluations of the SKLearn methods ===
    plot_training_curves(models, x_train, y_train, 'all_features.png', True)

    # === Plot different training evaluations ===
    plot_roc_auc(y_test, models_proba, 'roc_features_50.png')
    plot_evaluation_result(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix)
    plot_comparison(y_test, models_bool, ['Cohen kappa', 'f1', 'Precision', 'Recall', 'Number of yes'],
                    dnn_conf_matrix=dnn_confusion_matrix, filename='comparison_features_50_no_class_weight.png')
    plot_precision_recall(y_test, models_proba, 'pr_features_50.png')


if __name__ == "__main__":
    parse_config_file('configs/hyperparameters_50.gin')
    main()
