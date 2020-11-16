import time
import methods.utils as utils
from sklearn.model_selection import ShuffleSplit


class Classifier:
    """
    A generic Classifier class compatible with certain sklearn classifiers

    Parameters
    ----------
    cls : sklearn classifier object
        Classifier with its hyeperparameters set
    name : str
        The name of the classifier method used in the class instance


    Examples
    --------
    An instance of the classifier could then be used like this to instanciate
    a machine learning model from sklearn

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> classifier = Classifier(DecisionTreeClassifier, "Decision Tree")
    """

    def __init__(self, cls, name):
        self._model = cls  # Will never be trained, only used when plotting learning curve
        self._fitted_model = cls
        self._name = name
        self._short_name = "".join([w[0].lower() for w in name.split()]) if len(name.split()) > 1 else name.lower()

    def fit(self, x_train, y_train) -> None:
        """
        Instance method to train the model with given input parameters

        Parameters
        ----------
        x_train : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used
            as training input samples

        y_train : array
            An array of output sample values used during trianing
        """

        time_0 = time.time()
        print(f"{self._name} - start fitting...")
        self._fitted_model.fit(x_train, y_train)
        print(f"{self._name} - fit finished in {round(time.time() - time_0, 3)} s")

    def plot_learning_curves(self, x_train, y_train, plot_name) -> None:
        """
        Plots and saves the learning curves

        Plots are save in the directory 'methods/training_plots/' and will always train on
        an untrained model.

        Parameters
        ----------
        x_train : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used
            as training input samples

        y_train : array
            An array of output sample values used during trianing

        plot_name : str
            Name of the plot, required to have the file extension `.png`
        """

        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)  # TODO need to change this
        plot = utils.plot_learning_sklearn(self._model, self._name, x_train, y_train, cv=cv)
        plot_path = utils.save_training_plot(plot, f'{self._short_name}_{plot_name}')
        print(f'{self._name} -> Saved training plot in directory: "{plot_path}"')

    def predict(self, x_test):
        """
        Predicts the output from the given input on the fitted model

        Parameters
        ----------
        x_test : matrix{n_samples, n_features}
            Training input samples

        Returns
        -------
        y_pred_proba : list of float
            The predicted output values as probabilities: {0, 1}

        y_pred_bool : list of bool
            The predicted output values as boolean: [0, 1]
        """

        y_pred_proba = self._fitted_model.predict_proba(x_test)[:, 1]
        y_pred_bool = self._fitted_model.predict(x_test)
        return y_pred_proba, y_pred_bool

    def save_model(self, filename) -> None:
        """
        Saves the fitted model to the directory: 'saved_models'.

        Parameters
        ----------
        filename : string
            name of the file of the trained model,  required to have a `.sav` extension
        """

        utils.save_sklearn_model(filename, self._fitted_model, self._name)

    def load_model(self, filename) -> None:
        """
        Loads a model from file which will replace the fitted model.

        Parameters
        ----------
        filename : string
            Name of the file of the trained model, required to have a `.sav` extension
        """

        self._fitted_model = utils.load_sklearn_model(filename)
        print(f"{self._name} -> model loaded from: {filename}")


class SVC(Classifier):
    """
    Special case of the class Classifier, intended for the SVC method due to the inherent differences
    between the prediction method in other classifiers

    Parameters
    ----------

    cls : sklearn classifier object
        Classifier with its hyeperparameters set
    """

    def __init__(self, cls):
        super(SVC, self).__init__(cls, "SVC")

    def predict(self, x_test):
        """
        Predicts the output from the given input on the fitted model.

        Returns two equal predictions, with only boolean values. This is required
        since SVC does not predict probabilities that easily. Instead of just returning one
        array, it returns two so the main method of the project does not run into a RuntimeError

        Parameters
        ----------

        x_test : matrix{n_samples, n_features}
            Training input samples

        Returns
        -------
        y_pred_bool : list of bool
            The predicted output values as boolean: [0, 1]

        y_pred_bool : list of bool
            The predicted output values as boolean: [0, 1]
        """

        y_pred_bool = self._fitted_model.predict(x_test)
        return y_pred_bool, y_pred_bool
