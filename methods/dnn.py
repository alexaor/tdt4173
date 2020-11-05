import tensorflow as tf
from tensorflow.keras import layers
import gin.tf
import methods.utils as utils



"""
:param

:return


Creates a deep neural network classifier
"""


@gin.configurable(blacklist=['modelname'])
class DNN:
    def __init__(self, input_shape, dropout, optimizer_cls, metrics, loss, modelname="", **kwargs):
        self._kwargs = kwargs
        if len(modelname) > 0:
            self.model = utils.load_tf_model(modelname)
        else:
            self.model = self.create_model(input_shape, dropout)
            self.model.compile(optimizer=optimizer_cls, loss=loss, metrics=metrics)
        self.model.summary()

    def create_model(self, input_shape, dropout):
        return tf.keras.Sequential([
            # layers.Flatten(),
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(2)
        ])

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, self._kwargs)

    def save_model(self, modelname):
        utils.save_tf_model(modelname, self.model)
