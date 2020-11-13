from sklearn.neighbors import KNeighborsClassifier
import time
import methods.utils as utils
from sklearn.model_selection import ShuffleSplit
import gin


@gin.configurable
class KNN:
    """
    A class for K-Nearest Neighbors classifier, based on sklearn implementation.
    """
    def __init__(self, **kwargs):
        """
        Initialize the KNN classifier.

        :param kwargs:  hyperparameters to the classifier, which is being defined in configs/hyperparameters.gin
        """
        self._model = KNeighborsClassifier(**kwargs)  # Will never be trained, only used when plotting learning curves
        self._fitted_model = KNeighborsClassifier(**kwargs)

    def fit(self, x_train, y_train):
        """
        Fit the model with given input parameters, also prints the fitting time to terminal.

        :param x_train:      matrix{n_samples, n_features}, training input samples
        :param y_train:      array, training output samples
        """
        time_0 = time.time()
        print("KNN - start fitting...")
        self._fitted_model.fit(x_train, y_train)
        print(f"KNN - fit finished in {round(time.time() - time_0, 3)} s")

    def plot_learning_curves(self, x_train, y_train, plotname):
        """
        Plots the learning curves, and saves them in the directory 'methods/training_plots/'. It will always train on
        an untrained model.

        :param x_train:      matrix{n_samples, n_features}, training input samples
        :param y_train:      array, training output samples
        :param plotname:     str, name of the plot, need to have the file extension '.png'
        """
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)  # TODO need to change this
        plot = utils.plot_learning_sklearn(self._model, 'KNN', x_train, y_train, cv=cv)
        plotpath = utils.save_training_plot(plot, f'knn_{plotname}')
        print(f'KNN -> Saved training plot in directory: "{plotpath}"')

    def predict(self, x_test):
        """
        Predicts the output from the given input on the fitted model. Returns the output as probabilities and boolean.

        :param x_test:          matrix{n_samples, n_features}, training input samples
        :return y_pred_proba:   1darray, the predicted output values as probabilities: {0, 1}
        :return y_pred_bool:    1darray, the predicted output values as boolean: [0, 1]
        """
        y_pred_proba = self._fitted_model.predict_proba(x_test)[:, 1]
        y_pred_bool = self._fitted_model.predict(x_test)
        return y_pred_proba, y_pred_bool

    def save_model(self, filename):
        """
        Saves the fitted model to the directory: 'saved_models'.

        :param filename:    string, name of the file of the trained model, need to have file extension ".sav"
        """
        utils.save_sklearn_model(filename, self._fitted_model, 'KNN')

    def load_model(self, filename):
        """
        Loads model from file which will replace the fitted model.

        :param filename:    string, name of the file of the trained model, need to have file extension ".sav"
        """
        self._fitted_model = utils.load_sklearn_model(filename)
        print(f"KNN -> model loaded from: {filename}")
