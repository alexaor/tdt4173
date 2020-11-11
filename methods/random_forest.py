from sklearn.ensemble import RandomForestClassifier
import time
import methods.utils as utils
import gin

"""
:param x_train:      matrix{n_samples, n_features}, training input samples
:param y_train:      array, training output samples (1="yes" or 0="no")
:param x_test:       matrix{n_samples, n_features}, test input samples
:param filename:     string, name of the file of the trained model, need to have file extension ".sav"
:param **kwargs:     hyperparameters to the classifier, which is being defined in configs/hyperparameters.gin

:return y_pred: the predicted y values from the test set

Creates a random forest classifier with the given hyperparameters and trains the classifier. It returns
the predicted values after performing a test on the test input values.
"""


@gin.configurable
def random_forest(x_train, y_train, x_test, filename="", **kwargs):
    rf_classifier = RandomForestClassifier(**kwargs)
    time_0 = time.time()
    print("Random forest - start fitting...")
    for i in range(len(x_train)):
        rf_classifier.fit(x_train[i], y_train[i])
    print(f"Random forest - fit finished in {round(time.time() - time_0, 3)} s")
    y_pred = rf_classifier.predict(x_test)
    if len(filename) > 0:
        utils.save_sklearn_model(filename, rf_classifier)
    return y_pred
