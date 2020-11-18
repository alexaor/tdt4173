import pathlib
import pickle
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from colorama import Fore, Style
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import tensorflow as tf

from configs.project_settings import MODELS_PATH, PLOTS_PATH

MODEL_DIR = pathlib.Path(MODELS_PATH)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TRAINING_PLOT_DIR = pathlib.Path(PLOTS_PATH, "training_plots")
TRAINING_PLOT_DIR.mkdir(exist_ok=True, parents=True)


def save_sklearn_model(model_name, model, method):
    """
    Saves the model with given model_name in a directory specified in project settings

    Parameters
    ----------

    model_name : str
        name of the model
    method : str
        name of the method being saved
    model : sklearn.base.BaseEstimator
        sklearn class model to be saved
    """

    if model_name.endswith('.sav'):
        model_path = os.path.join(MODEL_DIR, model_name)
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"{method} -> Saving model to: '{model_path}'")
        pickle.dump(model, open(model_path, 'wb'))
    else:
        print(Fore.YELLOW + f"Warning: File extension unknown: {model_name.split('.')[-1]} \t-->\t should be .sav")
        print(Style.RESET_ALL)


def load_sklearn_model(model_name, method):
    """
    Loads and returns a saved sklearn model instance

    Returns the model saved with 'model_name', if the model does not exist it will exit the program with exit with exit
    code 1.

    Parameters
    ----------

    model_name : str
        Name of the model
    method : str
        name of the method being saved

    Returns
    -------
    loaded_model: sklearn.base.BaseEstimator
        klearn class model instance
    """

    model_path = os.path.join(MODEL_DIR, model_name)
    if os.path.isfile(model_path):
        print(f"{method} -> Loading model from: '{model_path}'")
        return pickle.load(open(model_path, 'rb'))
    else:
        print(Fore.RED + f"ERROR: Could not find the model: '{model_path}'")
        print(Style.RESET_ALL)
        exit(1)


def save_tf_model(model_name, model):
    if model_name.endswith('.h5') or model_name.endswith('.hdf5'):
        model_path = os.path.join(MODEL_DIR, model_name)
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"DNN -> Saving model to: '{model_path}'")
        model.save(model_path)
    else:
        print(Fore.YELLOW + f"Warning: Not correct file extension: {model_name} -> should be '.h5' or '.hdf5'")


def load_tf_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if os.path.isfile(model_path) and (model_name.endswith('.h5') or model_name.endswith('.hdf5')):
        print(f"DNN -> Model loaded from: '{model_path}'")
        return load_model(model_path, compile=False)
    else:
        if not model_name.endswith('.h5'):
            print(Fore.RED + f"ERROR: Not correct file extension: {model_name} -> should be '.h5' or '.hdf5'")
        else:
            print(Fore.RED + f"ERROR: Could not find the model: {model_path}")
        print(Style.RESET_ALL)
        exit(1)


def save_training_plot(fig, plotname):
    if not os.path.isdir(TRAINING_PLOT_DIR):
        print(Fore.RED + f'ERROR: Could not find directory: {TRAINING_PLOT_DIR}')
        print(Style.RESET_ALL)
        exit(1)
    if plotname.endswith('.png'):
        plot_path = os.path.join(TRAINING_PLOT_DIR, plotname)
        fig.savefig(plot_path)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{plotname.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'


def plot_tf_model(model, model_name):
    if not os.path.isdir(TRAINING_PLOT_DIR):
        print(Fore.RED + f'ERROR: Could not find directory: {TRAINING_PLOT_DIR}')
        print(Style.RESET_ALL)
        exit(1)
    if model_name.endswith('.png'):
        plot_path = os.path.join(TRAINING_PLOT_DIR, model_name)
        tf.keras.utils.plot_model(model, plot_path, show_shapes=True)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{model_name.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'


def plot_learning_sklearn(estimator, model_name, x, y, criterion=[], ylim=None, cv=5,
                          train_sizes=np.linspace(.1, 1.0, 5)):
    if len(estimator) != len(criterion) and criterion:
        print(Fore.YELLOW + f"Warning: The number of estimators vs number of model_names is not equal: {len(estimator)}"
                            f" vs. {len(criterion)}")
        return '-1'
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
