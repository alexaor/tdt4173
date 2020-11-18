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

model_output_dir = pathlib.Path(os.path.join('methods', 'saved_models'))
model_output_dir.mkdir(exist_ok=True, parents=True)

training_plot_dir = pathlib.Path(os.path.join('methods', 'training_plots'))
training_plot_dir.mkdir(exist_ok=True, parents=True)


"""
:param modelname:   string, name of the model
:param model:       sklearn class model

Saves the model with given modelname in the directory 'saved models'.
"""


def save_sklearn_model(modelname, model, method):
    if modelname.endswith('.sav'):
        modelpath = os.path.join(model_output_dir, modelname)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"{method} -> Saving model to: '{modelpath}'")
        pickle.dump(model, open(modelpath, 'wb'))
    else:
        print(Fore.YELLOW + f"Warning: File extension unknown: {modelname.split('.')[-1]} \t-->\t should be .sav")
        print(Style.RESET_ALL)


"""
:param modelname:   string, name of the model
:return             sklearn class model

Returns the model saved with 'modelname', if the model does not exist it will exit the program with exit with exit 
code 1.
"""


def load_sklearn_model(modelname, method):
    modelpath = os.path.join(model_output_dir, modelname)
    if os.path.isfile(modelpath):
        print(f"{method} -> Loading model from: '{modelpath}'")
        return pickle.load(open(modelpath, 'rb'))
    else:
        print(Fore.RED + f"ERROR: Could not find the model: '{modelpath}'")
        print(Style.RESET_ALL)
        exit(1)


def save_tf_model(modelname, model):
    if modelname.endswith('.h5') or modelname.endswith('.hdf5'):
        modelpath = os.path.join(model_output_dir, modelname)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"DNN -> Saving model to: '{modelpath}'")
        model.save(modelpath)
    else:
        print(Fore.YELLOW + f"Warning: Not correct file extension: {modelname} -> should be '.h5' or '.hdf5'")


def load_tf_model(modelname):
    modelpath = os.path.join(model_output_dir, modelname)
    if os.path.isfile(modelpath) and (modelname.endswith('.h5') or modelname.endswith('.hdf5')):
        print(f"DNN -> Model loaded from: '{modelpath}'")
        return load_model(modelpath, compile=False)
    else:
        if not modelname.endswith('.h5'):
            print(Fore.RED + f"ERROR: Not correct file extension: {modelname} -> should be '.h5' or '.hdf5'")
        else:
            print(Fore.RED + f"ERROR: Could not find the model: {modelpath}")
        print(Style.RESET_ALL)
        exit(1)


def save_training_plot(fig, plotname):
    if not os.path.isdir(training_plot_dir):
        print(Fore.RED + f'ERROR: Could not find directory: {training_plot_dir}')
        print(Style.RESET_ALL)
        exit(1)
    if plotname.endswith('.png'):
        plot_path = os.path.join(training_plot_dir, plotname)
        fig.savefig(plot_path)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{plotname.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'


def plot_tf_model(model, modelname):
    if not os.path.isdir(training_plot_dir):
        print(Fore.RED + f'ERROR: Could not find directory: {training_plot_dir}')
        print(Style.RESET_ALL)
        exit(1)
    if modelname.endswith('.png'):
        plot_path = os.path.join(training_plot_dir, modelname)
        tf.keras.utils.plot_model(model, plot_path, show_shapes=True)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{modelname.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'


def plot_learning_sklearn(estimator, modelname, x, y, criterion=[], ylim=None, cv=5,
                          train_sizes=np.linspace(.1, 1.0, 5)):
    if len(estimator) != len(criterion) and criterion:
        print(Fore.YELLOW + f"Warning: The number of estimators vs number of modelnames is not equal: {len(estimator)}"
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
            axes[0].set_title(f"Learning curve: {modelname}")
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
