import pathlib
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

from colorama import Fore, Style

from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator

from configs.project_settings import MODELS_PATH, PLOTS_PATH

MODEL_DIR = pathlib.Path(MODELS_PATH)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TRAINING_PLOT_DIR = pathlib.Path(PLOTS_PATH, "training_plots")
TRAINING_PLOT_DIR.mkdir(exist_ok=True, parents=True)


def save_sklearn_model(filename, model, method) -> None:
    """
    Saves the model with given model_name in a directory specified in project settings, model_name need to have
    extension '.sav'.

    Parameters
    ----------
    filename : string
        name of the model
    method : string
        name of the method being saved
    model : sklearn.base.BaseEstimator
        sklearn class model to be saved
    """

    if filename.endswith('.sav'):
        model_path = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"{method} -> Saving model to: '{model_path}'")
        pickle.dump(model, open(model_path, 'wb'))
    else:
        print(Fore.YELLOW + f"Warning: File extension unknown: {filename.split('.')[-1]} \t-->\t should be .sav")
        print(Style.RESET_ALL)


def load_sklearn_model(filename, method) -> BaseEstimator:
    """
    Loads and returns a saved sklearn model instance, filename need to have the extension: '.sav'.

    Returns the model saved with 'filename', if the model does not exist it will exit the program with exit
    code 1.

    Parameters
    ----------
    filename : string
        Name of the file to the model
    method : string
        name of the method being loaded

    Returns
    -------
    loaded_model: sklearn.base.BaseEstimator
        sklearn class model instance
    """

    model_path = os.path.join(MODEL_DIR, filename)
    if os.path.isfile(model_path):
        print(f"{method} -> Loading model from: '{model_path}'")
        return pickle.load(open(model_path, 'rb'))
    else:
        print(Fore.RED + f"ERROR: Could not find the model: '{model_path}'")
        print(Style.RESET_ALL)
        exit(1)


def save_tf_model(filename, model) -> None:
    """
    Saves the tensor flow model, need the file have '.h5' or '.hdf5' extension.

    Parameters
    ----------
    filename : string
        Name of the file to the model
    model : tensorflow.keras.Sequential
        The model that shall be saved
    """

    if filename.endswith('.h5') or filename.endswith('.hdf5'):
        model_path = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"DNN -> Saving model to: '{model_path}'")
        model.save(model_path)
    else:
        print(
            Fore.YELLOW + f"Warning: Did not save file. Not correct file extension: {filename} -> should be '.h5'"
                          f" or '.hdf5'")


def load_tf_model(filename) -> Sequential:
    """
    Loads and returns a saved tensorflow model instance, model_name need to have the extension: '.h5' or '.hdf5'.

    Returns the model saved with 'filename', if the model does not exist it will exit the program with exit
    code 1.

    Parameters
    ----------
    filename : string
        Name of the file where the model is saved

    Returns
    -------
    loaded_model : tensorflow.keras.Sequential
        Tensorflow model, this model is not compiled
    """

    model_path = os.path.join(MODEL_DIR, filename)
    if os.path.isfile(model_path) and (filename.endswith('.h5') or filename.endswith('.hdf5')):
        print(f"DNN -> Model loaded from: '{model_path}'")
        return load_model(model_path, compile=False)
    else:
        if not filename.endswith('.h5'):
            print(Fore.RED + f"ERROR: Not correct file extension: {filename} -> should be '.h5' or '.hdf5'")
        else:
            print(Fore.RED + f"ERROR: Could not find the model: {model_path}")
        print(Style.RESET_ALL)
        exit(1)


def save_training_plot(fig, filename) -> str:
    """
    Saves the figure with name 'filename' in directory: 'results/plots/training_plots'.

    Parameters
    ----------
    fig : matplotlib.pyplot
        The plot object that shall be saved
    filename : string
        Name of the file where the model is saved

    Returns
    -------
    path : string
        The path of where the plot is saved, if it did not manage to save the plot an error is printed out
        and the return string is '-1'.
    """

    if not os.path.isdir(TRAINING_PLOT_DIR):
        print(Fore.RED + f'ERROR: Could not find directory: {TRAINING_PLOT_DIR}')
        print(Style.RESET_ALL)
        exit(1)
    if filename.endswith('.png'):
        plot_path = os.path.join(TRAINING_PLOT_DIR, filename)
        fig.savefig(plot_path)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{filename.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'


def plot_learning_sklearn(estimator, model_name, x, y, criterion=[], ylim=None, cv=5,
                          train_sizes=np.linspace(.1, 1.0, 5)) -> plt:
    """
    Plot the learning curves for the given estimators given by using k-fold cross evaluation with the function
    'sklearn.model_selection.learning_curve' and returns the plot.

    Parameters
    ----------
    estimator : list of sklearn.base.BaseEstimator
        A list of estimators that shall be plotted against each other.
    model_name : string
        Name of the method
    x : numpy.ndarray
        The input values used for plotting the learning curve
    y : array
        The output values used for plotting the learning curve
    criterion : list of strings
        A list of criterion's which match the estimator
    ylim : tuple
        A tuple, (start, end),that sets the limitations on the y axis to the plots.
    cv : int
        Number of folds to make in the cross evaluation in, I.e., the k in k-fold cross evaluation
    train_sizes : array of floats
        List of floats, (0, 1], which represent the percentage of the dataset to do k-fold cross evaluation on

    Returns
    -------
    figure : matplotlib.pyplot
        The finished plot after the k-fold cross evaluation
    """

    if len(estimator) != len(criterion) and criterion:
        print(Fore.YELLOW + f"Warning: The number of estimators vs number of model_names is not equal: {len(estimator)}"
                            f" vs. {len(criterion)}")
        return plt.figure()
    else:
        plt.figure()
        color = [['r', 'g'], ['y', 'b']]
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for i in range(len(estimator)):
            if ylim is not None:
                axes[0].set_ylim(*ylim)
            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator[i], x, y, cv=cv,
                                                                                  train_sizes=train_sizes,
                                                                                  return_times=True)
            # Compute the mean along the x-axis
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            # Compute the standard deviation along the x-axis
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            # Plot learning curve
            axes[0].grid()
            # Plot the scores variation
            axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color=color[i][0])
            axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color=color[i][1])
            # Special case if there is two models to plot
            if criterion:
                axes[0].plot(train_sizes, train_scores_mean, 'o-', color=color[i][0],
                             label=f"Training score ({criterion[i]})")
                axes[0].plot(train_sizes, test_scores_mean, 'o-', color=color[i][1],
                             label=f"Cross-validation score ({criterion[i]})")
                axes[0].legend(loc="best")
            else:
                axes[0].plot(train_sizes, train_scores_mean, 'o-', color=color[i][0], label="Training score")
                axes[0].plot(train_sizes, test_scores_mean, 'o-', color=color[i][1], label="Cross-validation score")
                axes[0].legend(loc="best")
            axes[0].set_xlabel("Number of training samples")
            axes[0].set_ylabel("Score")
            axes[0].set_title(f"Learning curve: {model_name}")
            axes[0].grid(True)

            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
            if criterion:
                axes[1].plot(train_sizes, fit_times_mean, 'o-', label=criterion[i])
                axes[1].legend(loc="best")
            else:
                axes[1].plot(train_sizes, fit_times_mean, 'o-')
            axes[1].set_xlabel("Number of training samples")
            axes[1].set_ylabel("Time spent fitting")
            axes[1].set_title("Scalability of the model")
            axes[1].grid(True)

            # Plot fit_time vs score
            axes[2].grid()
            axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1)
            if criterion:
                axes[2].plot(fit_times_mean, test_scores_mean, 'o-', label=criterion[i])
                axes[2].legend(loc="best")
            else:
                axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
            axes[2].set_xlabel("Time spent fitting")
            axes[2].set_ylabel("Score")
            axes[2].set_title("Performance of the model")
            axes[2].grid(True)
        return fig
