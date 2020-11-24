from gin import parse_config_file

from evaluation.evaluate import (
    plot_precision_recall,
    plot_roc_auc,
    plot_evaluation_result,
    plot_comparison,
    print_evaluation
)
from numpy import log

from utils import (
    get_dataset,
    get_models,
    get_all_predictions,
    fit_all_models,
    plot_training_curves,
)

n_features = 'all'
def main():
    keys = ['Random Forest', 'DNN', 'Ada Boost', 'Decision Tree']
    dnn_confusion_matrix = None
    x_train, y_train = get_dataset(f'02_features_{n_features}_train.csv')
    x_test, y_test = get_dataset(f'02_features_{n_features}_test.csv')

    # Calculate initial bias for DNN
    pos = sum(y_train) + sum(y_test)
    neg = (len(y_train) + len(y_test)) - pos
    initial_bias = log([pos / neg])

    models = get_models((len(x_train[0]),), initial_bias, keys)
    models = fit_all_models(models, x_train, y_train)

    # === Loading models ===
    #models['DNN'].load_model(f'dnn_features_{n_features}.h5')
    #models['Ada Boost'].load_model(f'ab_tuned_features_{n_features}.sav')
    #models['Decision Tree'].load_model(f'dt_tuned_features_{n_features}.sav')
    #models['Random Forest'].load_model(f'rf_tuned_features_{n_features}.sav')

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
                    dnn_conf_matrix=dnn_confusion_matrix,
                    filename=f'comparison_features_{n_features}_no_class_weight.png')
    # plot_precision_recall(y_test, models_proba, 'pr_features_50.png')

    # === Saving models ===
    # models['Decision Tree'].save_model('dt_tuned_50_features.sav')
    # models['Ada Boost'].save_model('ab_tuned_all_features.sav')
    # models['Random Forest'].save_model('rf_tuned_50_features.sav')
    # models['DNN'].save_model('dnn_features_50.h5')


if __name__ == "__main__":
    # create_data_set('features_50', n_features=50)
    parse_config_file(f'configs/hyperparameters_{n_features}.gin')
    main()
