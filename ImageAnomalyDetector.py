from keras import Model, regularizers
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Softmax


class ImageAnomalyDetector(Model):
  def __init__(self):
    super(ImageAnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      Conv2D(64, (3, 3), activation='relu', padding='same',
                                  input_shape=(32, 32, 3), activity_regularizer=regularizers.l1(1e-6)),
      MaxPooling2D((2, 2), padding='same'),
      Dropout(0.2),
      Conv2D(32, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2), padding='same'),
      Dropout(0.2),
      Conv2D(16, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2), padding='same'),
      Dropout(0.2)
    ])

    self.decoder = tf.keras.Sequential([
      Conv2D(16, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Dropout(0.2),
      Conv2D(32, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Dropout(0.2),
      Conv2D(64, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Dropout(0.2),

      Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded