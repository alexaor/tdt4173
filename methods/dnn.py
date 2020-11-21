import matplotlib.pyplot as plt
import methods.utils as utils
import tensorflow as tf
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import layers, metrics

from typing import Tuple, Iterable


class DNN:
    """
    Class that creates a tensorflow Neural Network with respective functions.

    Parameters
    ----------
    input_shape : tuple of ints
        Shape of the samples, I.e., the number of features in the samples
    initial_bias : list of floats
        A list with only one float that sets the initial bias on the output layer
    dropout : float
        The dropout rate for the layers, injected from gin configuration file
    optimizer_cls : tensorflow optimizer object
        The optimizer to compile the model with, injected from gin configuration file
    loss : string
        Name of the loss function to the model, injected from gin configuration file
    **kwargs
        Keyword arguments are used to inject hyperparameters from gin the gin configuration file
    """

    def __init__(self, input_shape, initial_bias, dropout, optimizer_cls, loss, **kwargs):
        self._output_bias = initial_bias
        self._compile_para = {'optimizer': optimizer_cls, 'loss': loss, 'metrics': self._metrics}
        self._kwargs = kwargs
        self._metrics = [
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name='auc')
        ]
        self._unfitted_model = self._create_model(input_shape, dropout)  # Used in k-fold cross evaluation
        self.epoch_history = []
        self.model.summary()
        self.model = self._create_model(input_shape, dropout)
        self.model.compile(**self._compile_para)

    def _create_model(self, input_shape, dropout) -> tf.keras.Sequential:
        """
        Builds and returns the neural network.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input samples, should be on the format: (num_features,)
        dropout : float
            The dropout rate that will be set on the layers

        Returns
        -------
        tf.keras.Sequential
            The sequential model ready to be compiled and fitted
        """

        return tf.keras.Sequential([
            layers.Dense(units=40, activation='relu', input_shape=input_shape),
            layers.Dropout(dropout),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=1, activation='sigmoid',
                         bias_initializer=tf.keras.initializers.Constant(self._output_bias))
        ])

    def fit(self, x_train, y_train) -> None:
        """
        Instance method to train the model with given input parameters

        Parameters
        ----------
        x_train : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used as training input samples
        y_train : array
            An array of output sample values used during training
        """

        self.epoch_history = self.model.fit(x_train, y_train, **self._kwargs)

    def save_model(self, filename) -> None:
        """
        Saves the fitted model to the directory: 'saved_models'.

        Parameters
        ----------
        filename : string
            name of the file of the trained model, required to have a `.h5` or `.hdf5` extension
        """
        utils.save_tf_model(filename, self.model)

    def load_model(self, filename) -> None:
        """
        Loads a model from file which will replace model, and then compile the loaded model.

        Parameters
        ----------
        filename : string
            Name of the file of the trained model, required to have a `.h5` or `.hdf5` extension
        """

        self.model = utils.load_tf_model(filename)
        self.model.compile(**self._compile_para)

    def plot_training_evaluation(self, filename) -> None:
        """
        Plots the metrics, loss, precision recall and auc for each epoch in a plot, which is saved to the directory:
        'plots/training_plots'.

        Parameters
        ----------
        filename : string
            name of the file of the trained model, required to have a `.h5` or `.hdf5` extension
        """

        plt.plot(self.epoch_history.history['loss'])
        plt.plot(self.epoch_history.history['precision'])
        plt.plot(self.epoch_history.history['recall'])
        plt.plot(self.epoch_history.history['auc'])
        plt.title('Training evaluation')
        plt.xlabel('epoch')
        plt.legend(['Loss', 'Precision', 'recall', 'AUC'], loc='best')
        plot_path = utils.save_training_plot(plt, f'dnn_{filename}')
        print(f'\t DNN -> Saved training plot in directory: "{plot_path}"')

    def evaluate(self, x_test, y_true, threshold=0.5) -> Tuple[np.ndarray, Iterable]:
        """
        Predicts the output from the given input on the fitted model

        Parameters
        ----------
        x_test : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used as test input samples
        y_true : array
            An array of output sample values used during evaluating
        threshold : float
            A float that sets the threshold on if the prediction probability is 0 or 1

        Returns
        -------
        y_pred : list of float
            The predicted output values as probabilities: {0, 1}
        confusion_matrix : list of ints
            A list of the variables in the confusion matrix in the order: true negatives, false positives,
            false negatives and true positives
        """

        y_pred = self.model.predict(x_test, verbose=0)
        tp = tf.keras.metrics.TruePositives(thresholds=threshold)
        tp.update_state(y_true, y_pred)
        tp = tp.result().numpy()
        tn = tf.keras.metrics.TrueNegatives(thresholds=threshold)
        tn.update_state(y_true, y_pred)
        tn = tn.result().numpy()
        fp = tf.keras.metrics.FalsePositives(thresholds=threshold)
        fp.update_state(y_true, y_pred)
        fp = fp.result().numpy()
        fn = tf.keras.metrics.FalseNegatives(thresholds=threshold)
        fn.update_state(y_true, y_pred)
        fn = fn.result().numpy()
        confusion_matrix = [tn, fp, fn, tp]
        return y_pred, confusion_matrix

    def _fit_kfold(self, n_splits, dataset) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Do a k-fold cross evaluation and a normal fit operation on the dataset. Returns the result from the operations.

        Parameters
        ----------
        n_splits : int
            Number of splits to be done on the data, I.e., the k variable
        dataset : numpy.ndarray
            A dataset with the output variable still at the last row

        Returns
        -------
        cv_mean : list of floats
            The mean values of the metrics after performing k-fold cross evaluation, the order is: loss, precision,
            recall, auc
        cv_std : list of floats
            The variance of the metrics after performing k-fold cross evaluation, the order is: loss, precision,
            recall, auc
        scores : list of floats
            The scores to the predicting, after the model is trained normally, the order is: loss, precision, recall,
            auc
        """

        # Get the indexes to the different k folds
        kfold = KFold(n_splits=n_splits, shuffle=True)
        fold_no = 1

        # Create the input and output from the dataset
        x = dataset[:, :-1]
        y = dataset[:, -1]

        cv_scores = []

        # Perform k-fold cross evaluation
        for train, test in kfold.split(x, y):
            print(f"\t\t - Evaluating fold number {fold_no}")
            # Compile a new unfitted model with the given hyperparameters
            model = tf.keras.models.clone_model(self._unfitted_model)
            model.compile(**self._compile_para)
            # Fit model
            _ = model.fit(x[train], y[train], **self._kwargs, verbose=0)
            cv_score = model.evaluate(x[test], y[test], verbose=0)
            cv_scores.append(cv_score)
            fold_no += 1

        # Perform normal training
        model = tf.keras.models.clone_model(self._unfitted_model)
        model.compile(**self._compile_para)
        train, test = train_test_split(dataset, train_size=0.8)
        _ = model.fit(train[:, :-1], train[:, -1], **self._kwargs, verbose=0)
        scores = model.evaluate(test[:, :-1], test[:, -1], verbose=0)

        cv_mean = []
        cv_std = []

        # Calculate mean and variance for the score
        for i in range(4):
            cv_mean.append(np.mean([cv_scores[j][i] for j in range(n_splits)]))
            cv_std.append(np.std([cv_scores[j][i] for j in range(n_splits)]))

        return cv_mean, cv_std, scores

    def plot_cross_evaluation(self, n_splits, x_train, y_train, x_test, y_test, filename,
                              train_sizes=np.linspace(.1, 1.0, 5)) -> None:
        """
        Perform k-fold cross evaluation on the dataset, and saves the plot in the directory: 'plots/training_plots'.

        Parameters
        ----------
        n_splits : int
            Number of splits to be done on the dataset in k-fold evaluation, I.e., the k variable
        x_train : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used to merge the dataset
        y_train : array
            An array of output sample values used to merge the dataset
        x_test : numpy.ndarray
            A numpy.ndarray matrix consisting of n_samples and n_features, used to merge the dataset
        y_test : array
            An array of output sample values used to merge the dataset
        filename : string
            name of the file of the trained model, required to have a `.png` extension
        train_sizes : array of floats
            List of floats, (0, 1], which represent the percentage of the dataset to do k-fold cross evaluation on
        """

        # Merging the dataset parts to one parameter
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        dataset = np.zeros((len(x), len(x[0]) + 1), float)
        dataset[:, :-1] = x
        dataset[:, -1] = y

        scores = {'Loss': [], 'Precision': [], 'Recall': [], 'AUC': []}
        cv_mean = {'Loss': [], 'Precision': [], 'Recall': [], 'AUC': []}
        cv_std = {'Loss': [], 'Precision': [], 'Recall': [], 'AUC': []}

        # Start cross evaluation
        for size in train_sizes:
            if size != 1.0:
                train, _ = train_test_split(dataset, train_size=size)
            else:
                train = dataset

            print(f"DNN -> Cross evaluating on dataset size: {len(train)}")
            mean, std, score = self._fit_kfold(n_splits, train)

            cv_mean['Loss'].append(mean[0])
            cv_mean['Precision'].append(mean[1])
            cv_mean['Recall'].append(mean[2])
            cv_mean['AUC'].append(mean[3])
            cv_std['Loss'].append(std[0])
            cv_std['Precision'].append(std[1])
            cv_std['Recall'].append(std[2])
            cv_std['AUC'].append(std[3])
            scores['Loss'].append(score[0])
            scores['Precision'].append(score[1])
            scores['Recall'].append(score[2])
            scores['AUC'].append(score[2])

        # Plot figure
        sizes = [int(len(x) * size) for size in train_sizes]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        index = 0
        for metric in cv_mean.keys():
            axes[index].fill_between(sizes,
                                     [cv_mean[metric][i] - cv_std[metric][i] for i in range(len(cv_mean[metric]))],
                                     [cv_mean[metric][i] + cv_std[metric][i] for i in range(len(cv_mean[metric]))],
                                     alpha=0.1, color='g')
            axes[index].plot(sizes, cv_mean[metric], '-o', color='g', label='Cross validation score')
            axes[index].plot(sizes, scores[metric], '-o', color='r', label='Training score')
            axes[index].set_xlabel("Number of training samples")
            axes[index].set_ylabel(metric)
            axes[index].set_title(f"DNN - {metric}")
            axes[index].grid(True)
            axes[index].legend(loc="best")

            index += 1

        plot_path = utils.save_training_plot(fig, f'dnn_cv_{filename}')
        print(f'\t DNN -> Saved training plot in directory: "{plot_path}"')
