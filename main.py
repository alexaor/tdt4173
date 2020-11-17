import methods
import evaluate.evaluate as evaluate
import gin
import methods.utils as utils
from preprocess.preprocessor import create_data_set


def get_models(input_shape, keys):
    models = {}
    if 'Random Forest' in keys:
        models['Random Forest'] = methods.models.random_forest()
    if 'Ada Boost' in keys:
        models['Ada Boost'] = methods.models.ada_boost()
    if 'Decision Tree' in keys:
        models['Decision Tree'] = methods.models.decision_tree()
    if 'KNN' in keys:
        models['KNN'] = methods.models.knn()
    if 'DNN' in keys:
        models['DNN'] = methods.dnn.DNN(input_shape=input_shape)
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
    keys = ['Random Forest']
    dnn_confusion_matrix = None
    x_train, y_train = utils.get_dataset('Features_50_train.csv')
    x_test, y_test = utils.get_dataset('Features_50_test.csv')
    models = fit_all_models(get_models((len(x_train[0]),), keys), x_train, y_train)
    models_proba, models_bool = get_all_predictions(models, x_test)
    if 'DNN' in keys:
        models_proba['DNN'], dnn_confusion_matrix = models['DNN'].evaluate(x_test, y_test)

    print('\n\n================ Evaluation ================\n')
    # models['DNN'].plot_model('DNN_network.png')
    plot_training_curves(models, x_train, y_train, '50features.png', True)
    evaluate.print_evaluation(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix)
    evaluate.plot_roc_auc(y_test, models_proba)
    # evaluate.plot_evaluation_result(y_test, models_bool, dnn_conf_matrix=dnn_confusion_matrix)
    # evaluate.plot_comparison(y_test, models_bool, ['accuracy', 'f1', 'Specificity', 'False positive rate'],
    #                         dnn_conf_matrix=dnn_confusion_matrix)

    # models['Decision Tree'].save_model('dt_tuned_50Features.sav')
    # models['Ada Boost'].save_model('ab_tuned_50Features.sav')
    # odels['Random Forest'].save_model('rf_tuned_50Features.sav')
    # models['DNN'].save_model('dnn_tuned_50Features')

if __name__ == "__main__":
    #create_data_set('Features_50', n_features=50)
    gin.parse_config_file('configs/hyperparameters50.gin')
    main()
