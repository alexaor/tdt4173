import time
import methods.utils as utils
from sklearn.model_selection import ShuffleSplit
from typing import Tuple
import numpy as np

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

    def __init__(self, cls, name, **kwargs):
        self._classifier = cls
        self._model = cls(**kwargs)
        self._name = name
        self._short_name = "".join([w[0].lower() for w in name.split()]) if len(name.split()) > 1 else name.lower()
        self._kwargs = kwargs
        if 'criterion' in kwargs.keys():
            self._criterion = kwargs['criterion']

    def fit(self, x_train, y_train) -> None:
        """
        Instance method to train the model with given input parameters

        Parameters
        ----------
        x_train : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used
            as training input samples
        y_train : array
            An array of output sample values used during training
        """

        time_0 = time.time()
        print(f"{self._name} - start fitting...")
        self._model.fit(x_train, y_train)
        print(f"{self._name} - fit finished in {round(time.time() - time_0, 3)} s")

    def plot_learning_curves(self, x_train, y_train, plot_name, compare_criterion=False) -> None:
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
            An array of output sample values used during training
        plot_name : str
            Name of the plot, required to have the file extension `.png`
        compare_criterion :  bool
            If the learning curve should compare the two different criterion or not
        """

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)  # TODO need to change this
        criterions = ['gini', 'entropy']
        if compare_criterion:
            models = []
            for criterion in criterions:
                self._kwargs.pop('criterion')
                self._kwargs['criterion'] = criterion
                models.append(self._classifier(**self._kwargs))
            plot = utils.plot_learning_sklearn(models, self._name, x_train, y_train, criterion=criterions, cv=cv)
        else:
            plot = utils.plot_learning_sklearn([self._classifier(**self._kwargs)], self._name, x_train, y_train, cv=cv)
        plot_path = utils.save_training_plot(plot, f'{self._short_name}_{plot_name}')
        print(f'{self._name} -> Saved training plot in directory: "{plot_path}"')

    def predict(self, x_test) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the output from the given input on the fitted model

        Parameters
        ----------
        x_test : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used
            as test input samples

        Returns
        -------
        y_pred_proba : list of float
            The predicted output values as probabilities: {0, 1}
        y_pred_bool : list of bool
            The predicted output values as boolean: [0, 1]
        """

        y_pred_proba = self._model.predict_proba(x_test)[:, 1]
        y_pred_bool = self._model.predict(x_test)
        return y_pred_proba, y_pred_bool

    def save_model(self, filename) -> None:
        """
        Saves the fitted model to the directory: 'saved_models'.

        Parameters
        ----------
        filename : string
            name of the file of the trained model,  required to have a `.sav` extension
        """

        utils.save_sklearn_model(filename, self._model, self._name)

    def load_model(self, filename) -> None:
        """
        Loads a model from file which will replace the fitted model.

        Parameters
        ----------
        filename : string
            Name of the file of the trained model, required to have a `.sav` extension
        """

        self._model = utils.load_sklearn_model(filename, self._name)
