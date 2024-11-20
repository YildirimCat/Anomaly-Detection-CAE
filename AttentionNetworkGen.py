import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply

@keras.saving.register_keras_serializable(package="MyLayers")
class AttentionBlock(Layer):
    def __init__(self, channels, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.channels = channels
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(channels // 8, activation='relu')
        self.dense2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape((1, 1, channels))
        self.multiply = Multiply()

    def call(self, inputs):
        attention = self.global_avg_pool(inputs)
        attention = self.dense1(attention)
        attention = self.dense2(attention)
        attention = self.reshape(attention)
        return self.multiply([inputs, attention])

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config