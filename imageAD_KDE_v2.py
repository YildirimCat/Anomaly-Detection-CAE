import random
import keras.callbacks
import numpy as np
from PIL import Image
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import f1_score, roc_curve, auc, classification_report

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Sequential, load_model

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Calculate KDE using sklearn
from sklearn.neighbors import KernelDensity

import os
import ImageAnomalyDetector
import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
tf.get_logger().setLevel('INFO')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Tensorflow version: ", tf.__version__)

print('')

data_dir = './dataset/3rd_test/4th_test/txt'

# Utils.generate_bearing_data_csv(data_dir)

merged_data = pd.DataFrame()
merged_data = merged_data._append(pd.read_csv('./Averaged_Dataset_BearingTest_3.csv', index_col=0))
print(merged_data.head())
merged_data.plot(figsize=(12,6))
plt.show()