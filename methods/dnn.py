import tensorflow as tf
from tensorflow.keras import layers
import gin.tf
import methods.utils as utils
import matplotlib.pyplot as plt


@gin.configurable(blacklist=['modelname'])
class DNN:
    def __init__(self, input_shape, dropout, optimizer_cls, metrics, loss, modelname="", **kwargs):
        self._kwargs = kwargs
        self.epoch_history = []
        if len(modelname) > 0:
            self.model = utils.load_tf_model(modelname)
        else:
            self.model = self.create_model(input_shape, dropout)
            self.model.compile(optimizer=optimizer_cls, loss=loss, metrics=metrics)
        self.model.summary()


    def create_model(self, input_shape, dropout):
        return tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            layers.Dense(units=20, activation='relu'),
            layers.Dense(units=40, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(units=1, activation='sigmoid')
        ])

    def fit(self, x_train, y_train):
        self.epoch_history = self.model.fit(x_train, y_train, **self._kwargs)

    def save_model(self, modelname):
        utils.save_tf_model(modelname, self.model)

    def plot_accuracy(self, filename):
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
