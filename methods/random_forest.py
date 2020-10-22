from sklearn.ensemble import RandomForestClassifier
import time
import gin


"""
:param x_train:       matrix{n_samples, n_features}, training input samples
:param y_train:       array, training output samples ("yes" or "no")
:param x_test:        matrix{n_samples, n_features}, test input samples
:param **kwargs:      hyperparameters to the classifier, which is being defined in configs/random_forest.gin

:return y_pred: the predicted y values from the test set

Creates an random forest classifier with the given hyperparameters and trains the classifier. It returns
the predicted values after performing a test on the test input values.
"""


# TODO vil vi bruke gin? er en nice måte og sette hyperparametere på, kan også bare bruke en config fil, ikke en for hver
@gin.configurable
def random_forest(x_train, y_train, x_test, **kwargs):
    rf_classifier = RandomForestClassifier(**kwargs)
    time_0 = time.time()
    print("Random forest - start fitting...")
    rf_classifier.fit(x_train, y_train)
    print(f"Random forest - fit finished in {round(time.time() - time_0, 3)} s")
    y_pred = rf_classifier.predict(x_test)
    return y_pred

# TODO vil vi ha det som står under? Er det samme som over bare med klasse definisjon istedenfor
"""
def random_forest(x_train, y_train, x_test):
    rf = RandomForest()
    rf.train(x_train, y_train)
    return rf.predict(x_test)


@gin.configurable
class RandomForest:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, x_train, y_train):
        time_0 = time.time()
        print("Random forest - start fitting...")
        self.model.fit(x_train, y_train)
        print(f"Random forest - fit finished in {round(time.time() - time_0, 3)} s")

    def predict(self, x_test):
        return self.model.predict(x_test)
"""