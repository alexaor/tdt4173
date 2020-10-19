
from sklearn.tree import DecisionTreeClassifier
import time


"""
:param x_train:      matrix{n_samples, n_features}, training input samples
:param y_train:      array, training output samples ("yes" or "no")
:param x_test:       matrix{n_samples, n_features}, test input samples
:param max_depth:    int, the maximum depth of the tree, default at None
:param splitter:     {'best', 'random'}, strategy on how the node split is chosen
:param max_features: {'None', 'auto', 'sqrt', 'log2'}, the number of features to consider when splitting

:return y_pred: the predicted y values from the test set

Creates a decision tree classifier with the given hyperparameters and trains the classifier. I returns
the predicted values after performing a test on the test input values.
"""


def decision_tree(x_train, y_train, x_test, max_depth=None, splitter='best', max_features=None):
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, splitter=splitter, max_features=max_features)
    time_0 = time.time()
    print("Decision tree - start fitting...")
    dt_classifier.fit(x_train, y_train)
    print("Decision tree - fit finished in {} s".format(round(time.time() - time_0, 3)))
    y_pred = dt_classifier.predict(x_test)
    return y_pred

