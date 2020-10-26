from sklearn.neighbors import KNeighborsClassifier
import time
import pickle
import gin

"""
:param x_train:      matrix{n_samples, n_features}, training input samples
:param y_train:      array, training output samples (1="yes" or 0="no")
:param x_test:       matrix{n_samples, n_features}, test input samples
:param filename:     string, name of the file of the trained model, need to have file extension ".sav"
:param **kwargs:     hyperparameters to the classifier, which is being defined in configs/hyperparameters.gin

:return y_pred: the predicted y values from the test set

Creates a k-nearest neighbors classifier with the given hyperparameters and trains the classifier. It returns
the predicted values after performing a test on the test input values. There is also possible to save the trained model
for future use.
"""


@gin.configurable
def knn(x_train, y_train, x_test, filename="", **kwargs):
    knn_classifier = KNeighborsClassifier(**kwargs)
    time_0 = time.time()
    print("KNN - start fitting...")
    knn_classifier.fit(x_train, y_train)
    print(f"KNN - fit finished in {round(time.time() - time_0, 3)} s")
    y_pred = knn_classifier.predict(x_test)
    if len(filename) > 0:
        if filename.endswith('.sav'):
            pickle.dump(knn_classifier, open(filename, 'wb'))
            print(f"Model saved as: {filename}")
        else:
            print(f"file extension unknown: {filename.split('.')[-1]} \t\t-->\t did not save file")
    return y_pred