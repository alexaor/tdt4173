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

model_output_dir = pathlib.Path(os.path.join('methods','saved_models'))
model_output_dir.mkdir(exist_ok=True, parents=True)

dataset_path = pathlib.Path('preprocess/datasets')

training_plot_dir = pathlib.Path(os.path.join('methods', 'training_plots'))
training_plot_dir.mkdir(exist_ok=True, parents=True)

"""
:param modelpath:   string, path to the model
:return             bool, true if the file exist and false if it don't

Checks if there is a file in at the end of the modelpath
"""
def model_exist(modelpath):
    return os.path.isfile(modelpath)


"""
:param modelname:   string, name of the model
:param model:       sklearn class model

Saves the model with given modelname in the directory 'saved models'.
"""
def save_sklearn_model(modelname, model, method):
    if modelname.endswith('.sav'):
        modelpath = os.path.join(model_output_dir, modelname)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"{method} -> Saving model to: {modelpath}")
        pickle.dump(model, open(modelname, 'wb'))
    else:
        print(Fore.YELLOW + f"Warning: File extension unknown: {modelname.split('.')[-1]} \t-->\t should be .sav")
        print(Style.RESET_ALL)


"""
:param modelname:   string, name of the model
:return             sklearn class model

Returns the model saved with 'modelname', if the model does not exist it will exit the program with exit with exit 
code 1.
"""
def load_sklearn_model(modelname):
    modelpath = os.path.join(model_output_dir, modelname)
    if model_exist(modelpath):
        return pickle.load(open(modelpath, 'rb'))
    else:
        print(Fore.RED + f"ERROR: Could not find the model: {modelpath}")
        print(Style.RESET_ALL)
        exit(1)


def save_tf_model(modelname, model):
    modelpath = os.path.join(model_output_dir, modelname)
    os.makedirs(model_output_dir, exist_ok=True)
    print("DNN -> Saving model to: ", modelpath)
    model.save(modelpath)


def load_tf_model(modelname):
    modelpath = os.path.join(model_output_dir, modelname)
    if os.path.isdir(modelpath):
        return load_model(modelpath)
    else:
        print(Fore.RED + f"ERROR: Could not find the directory: {modelpath}")
        print(Style.RESET_ALL)
        exit(1)
        
        
def get_dataset(filename):
    dataset = pd.read_csv(os.path.join(dataset_path, filename))
    x_train = dataset.iloc[:, :-3].values
    y_train = dataset.iloc[:, -1].values
    return x_train, y_train


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
        tf.keras.utils.plot_model(model, modelname, show_shapes=True)
        return plot_path
    else:
        print(Fore.YELLOW + f'Warning: File extension wrong: ".{modelname.split(".")[-1]}" \t--> should be ".png"')
        print(Style.RESET_ALL)
        return '-1'



def plot_learning_sklearn(estimator, modelname, x, y, axes=None, ylim=None, cv=5, n_jobs=None,
                          train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].set_title(modelname)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator, x, y, cv=cv,
                                                    n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt
