from sklearn.ensemble import AdaBoostClassifier
import time
import pickle


"""
:param x_train:       matrix{n_samples, n_features}, training input samples
:param y_train:       array, training output samples ("yes" or "no")
:param x_test:        matrix{n_samples, n_features}, test input samples
:param save_model:    bool, true if you want to save the trained classifier model
:param filename:      string, name of the file of the classifier, need to be of type ".sav"
:param learning_rate: float, shrinks the contribution of each classifier
:n_estimators:        int, max number of estimators (trees in the forest) at which boosting is terminated 

:return y_pred: the predicted y values from the test set

Creates an AdaBoost classifier with the given hyperparameters and trains the classifier. It returns
the predicted values after performing a test on the test input values. It is using the base estimator, which is
decision tree with a max depth of 1.
"""

# TODO Skal vi lagre modellene, eller skal vi også returnere modellen så vi kan lagre den i main?
def adaBoost(x_train, y_train, x_test, save_model=False, filename="", learning_rate=1, n_estimators=50):
    ab_classifier = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    time_0 = time.time()
    print("AdaBoost - start fitting...")
    ab_classifier.fit(x_train, y_train)
    print(f"AdaBoost - fit finished in {round(time.time() - time_0, 3)} s")
    y_pred = ab_classifier.predict(x_test)
    if save_model and len(filename) > 0:
        pickle.dump(ab_classifier, open(filename, 'wb'))
    return y_pred
