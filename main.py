import methods.random_forest as rf
import methods.ada_boost as ab
import methods.decision_tree as dt
import methods.svc as svc
import methods.knn as knn
import methods.dnn as dnn
from sklearn.metrics import accuracy_score
import evaluate.evaluate as ev
import gin
import methods.utils as utils


if __name__ == "__main__":
    gin.parse_config_file('configs/hyperparameters.gin')
    print("started main")

    x_train, y_train = utils.get_dataset('normalized_training_set.csv')
    x_test, y_test = utils.get_dataset('normalized_test_set.csv')
    """
    model = dnn.DNN()
    model.fit_model(x_train, y_train)
    loss = model.model.evaluate(x_test, y_test)
    print(model.model.metrics_names)
    print(loss)
    """
    methods = {}
    methods['Random Forest'] = rf.random_forest([x_train], [y_train], x_test)
    methods['Ada Boost'] = ab.ada_boost([x_train], [y_train], x_test)
    methods['Decision Tree'] = dt.decision_tree([x_train], [y_train], x_test)
    methods['SVC'] = svc.svc([x_train], [y_train], x_test)
    methods['KNN'] = knn.knn([x_train], [y_train], x_test)
    # methods['DNN'] = model.model.predict(x_test)

    print('\n\n================ Evaluation ================\n')
    ev.sklearn_print_evaluation(y_test, methods)
    ev.sklearn_auc(y_test, methods)

    """
    index = 0
    score = []
    for p in pred:
        score.append(accuracy_score(p, y_test))
    for method in methods:
        print(f'{method}:\t{score[index]}')
        index += 1
    """
