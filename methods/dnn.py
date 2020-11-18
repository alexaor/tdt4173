import tensorflow as tf
from tensorflow.keras import layers
import methods.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split


class DNN:
    def __init__(self, input_shape, dropout, optimizer_cls, metrics, loss, **kwargs):
        self._kwargs = kwargs
        self._compile_para = {'optimizer': optimizer_cls, 'loss': loss, 'metrics': metrics}
        self._unfitted_model = self.create_model(input_shape, dropout)
        self.epoch_history = []
        self.model = self.create_model(input_shape, dropout)
        self.model.compile(**self._compile_para)
        self.model.summary()


    def create_model(self, input_shape, dropout):
        return tf.keras.Sequential([

            layers.Dense(units=64, activation='relu', input_shape=input_shape),
            layers.Dense(units=20, activation='relu'),
            layers.Dense(units=10, activation='relu'),
            layers.Dense(units=4, activation='relu'),
            layers.Dense(units=1, activation='sigmoid')
        ])

    def fit(self, x_train, y_train):
        self.epoch_history = self.model.fit(x_train, y_train, **self._kwargs)

    def save_model(self, modelname):
        utils.save_tf_model(modelname, self.model)

    def load_model(self, modelname):
        self.model = utils.load_tf_model(modelname)
        self.model.compile(**self._compile_para)

    def plot_accuracy(self, filename):
        k = self.epoch_history.history['accuracy']
        plt.plot(self.epoch_history.history['accuracy'])
        plt.plot(self.epoch_history.history['loss'])
        plt.plot(self.epoch_history.history['mse'])
        plt.title('Training evaluation')
        plt.xlabel('epoch')
        plt.legend(['Accuracy', 'Loss', 'mse'], loc='best')
        plotpath = utils.save_training_plot(plt, f'dnn_{filename}')
        print(f'\t DNN -> Saved training plot in directory: "{plotpath}"')

    def evaluate(self, x_test, y_true, threshold=0.5):
        y_pred = self.model.predict(x_test, verbose=1)
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

    def plot_model(self, filename):
        plotpath = utils.plot_tf_model(self.model, filename)
        print(f'\t DNN -> Saved model plot in directory: "{plotpath}"')

    def _learning_evaluation(self, n_splits, dataset):
        kfold = KFold(n_splits=n_splits, shuffle=True)
        fold_no = 1

        x = dataset[:, :-1]
        y = dataset[:, -1]

        cv_scores = []

        # Perform k-fold cross evaluation
        for train, test in kfold.split(x, y):
            # Compile a new unfitted model with the given hyperparameters
            model = tf.keras.models.clone_model(self._unfitted_model)
            model.compile(**self._compile_para)
            # Fit model
            _ = model.fit(x[train], y[train], **self._kwargs)
            cv_score = model.evaluate(x[test], y[test], verbose=0)
            cv_scores.append(cv_score)
            fold_no += 1

        # Perform normal training
        model = tf.keras.models.clone_model(self._unfitted_model)
        model.compile(**self._compile_para)
        train, test = train_test_split(dataset, train_size=0.8)
        _ = model.fit(train[:, :-1], train[:, -1], **self._kwargs)
        scores = model.evaluate(test[:, :-1], test[:, -1], verbose=0)
        cv_mean = []
        cv_std = []

        # Calculate mean and variance for the score
        for i in range(3):
            cv_mean.append(np.mean([cv_scores[j][i] for j in range(n_splits)]))
            cv_std.append(np.std([cv_scores[j][i] for j in range(n_splits)]))

        return cv_mean, cv_std, scores

    def plot_cross_evaluation(self, n_splits, x_train, y_train, x_test, y_test, filename,
                              train_sizes=np.linspace(.1, 1.0, 5)):
        # Merging inputs and targets
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        dataset = np.zeros((len(x), len(x[0]) + 1), float)
        dataset[:, :-1] = x
        dataset[:, -1] = y

        scores = {'Loss': [], 'Accuracy': [], 'MSE': []}
        cv_mean = {'Loss': [], 'Accuracy': [], 'MSE': []}
        cv_std = {'Loss': [], 'Accuracy': [], 'MSE': []}

        for size in train_sizes:
            if size != 1.0:
                train, _ = train_test_split(dataset, train_size=size)
            else:
                train = dataset

            mean, std, score = self._learning_evaluation(n_splits, train)

            cv_mean['Loss'].append(mean[0])
            cv_mean['Accuracy'].append(mean[1])
            cv_mean['MSE'].append(mean[2])
            cv_std['Loss'].append(std[0])
            cv_std['Accuracy'].append(std[1])
            cv_std['MSE'].append(std[2])
            scores['Loss'].append(score[0])
            scores['Accuracy'].append(score[1])
            scores['MSE'].append(score[2])

        # Plot figure
        sizes = [int(len(x)*size) for size in train_sizes]
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
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

        plotpath = utils.save_training_plot(fig, f'dnn_cv_{filename}')
        print(f'\t DNN -> Saved training plot in directory: "{plotpath}"')
