import time
import methods.utils as utils
from sklearn.model_selection import ShuffleSplit


class Classifier:
    """
    A class that can be used by sklearn's different classifiers, which gives us the functions we need.
    """
    def __init__(self, cls, name):
        """
        Initialize the classifier with the correct sklearn classifier.

        :param cls:         sklearn classifier object, classifier with its hyeperparameters set
        :param name:        str, name of the classifier
        """
        self._model = cls  # Will never be trained, only used when plotting learning curve
        self._fitted_model = cls
        self._name = name
        self._short_name = "".join([w[0].lower() for w in name.split()]) if len(name.split()) > 1 else name.lower()

    def fit(self, x_train, y_train):
        """
        Fit the model with given input parameters, also prints the fitting time to terminal.

        :param x_train:      matrix{n_samples, n_features}, training input samples
        :param y_train:      array, training output samples
        """
        time_0 = time.time()
        print(f"{self._name} - start fitting...")
        self._fitted_model.fit(x_train, y_train)
        print(f"{self._name} - fit finished in {round(time.time() - time_0, 3)} s")

    def plot_learning_curves(self, x_train, y_train, plotname):
        """
        Plots the learning curves, and saves them in the directory 'methods/training_plots/'. It will always train on
        an untrained model.

        :param x_train:      matrix{n_samples, n_features}, training input samples
        :param y_train:      array, training output samples
        :param plotname:     str, name of the plot, need to have the file extension '.png'
        """
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)  # TODO need to change this
        plot = utils.plot_learning_sklearn(self._model, self._name, x_train, y_train, cv=cv)
        plotpath = utils.save_training_plot(plot, f'{self._short_name}_{plotname}')
        print(f'{self._name} -> Saved training plot in directory: "{plotpath}"')

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
        utils.save_sklearn_model(filename, self._fitted_model, self._name)

    def load_model(self, filename):
        """
        Loads model from file which will replace the fitted model.

        :param filename:    string, name of the file of the trained model, need to have file extension ".sav"
        """
        self._fitted_model = utils.load_sklearn_model(filename)
        print(f"{self._name} -> model loaded from: {filename}")


class SVC(Classifier):
    """
    Special case of the class Classifier, intended for the SVC method. Since that method need to have a different
    predict function.
    """
    def __init__(self, cls, name):
        """
        Only initialise the inherited class.

        :param cls:         sklearn classifier object, classifier with its hyeperparameters set
        :param name:        str, name of the classifier
        """
        super(SVC, self).__init__(cls, name)

    def predict(self, x_test):
        """
        Predicts the output from the given input on the fitted model. Returns two equal predictions, with only bollean
        values. This is needed since SVC don't predict probabilities that easily. Instead of just returning one
        array, it returns two so other functions don't need a special case for the SVC.

        :param x_test:          matrix{n_samples, n_features}, training input samples
        :return y_pred_bool:    1darray, the predicted output values as probabilities: [0, 1]
        :return y_pred_bool:    1darray, the predicted output values as boolean: [0, 1]
        """
        y_pred_bool = self._fitted_model.predict(x_test)
        return y_pred_bool, y_pred_bool
