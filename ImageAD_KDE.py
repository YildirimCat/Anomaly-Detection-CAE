"""
@Author:    Yıldırım Çat
@Date:      18.08.2024
@Goal:      Zaman serisi verilerinin imgeye dönüştürülmesinden sonra CAE modeliyle anomali tespiti yapılması
@Method:    İstatistiksel Profil Çıkarma Yaklaşımı (Statistical Profiling Approach)
@Sources:   https://www.tensorflow.org/tutorials/generative/autoencoder
"""
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
import AttentionNetworkGen

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
merged_data = pd.DataFrame()
merged_data = merged_data._append(pd.read_csv('./Averaged_Bearing_Dataset_test_2.csv', index_col=0))
print(merged_data.head())
merged_data.plot(figsize=(12,6))
plt.show()


# Bearing Test1 Train: İlk 400 örnek, Test: Tüm örnekler
train = merged_data['2004-02-12 10:32:39':'2004-02-15 05:02:39']
test = merged_data[:]

print('Training dataset shape: ', train.shape)
print('Test dataset shape: ', test.shape)

print('')
print(merged_data[['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']].describe())

# Labelling data classes
# [0-531]   --> 1
# [532-983] --> 0
positive_label = np.ones((532, 1))
train_label = np.ones((400, 1))
negative_label = np.zeros((452, 1))
labels = np.concatenate((positive_label, negative_label))
labelled_data = np.append(merged_data, labels, axis = 1)


# Test for the first file
sample_dataset = pd.read_csv('./dataset/2nd_test/2nd_test/2004.02.12.10.32.39', sep='\t')
ax = sample_dataset.plot(figsize = (12,6), title= "Rulman Titreşim" , legend = True)
ax.set(xlabel="çevirim(n)", ylabel="Titreşim/İvme(g)")
plt.show()

SIZE = 32
EPOCHS = 100
BATCH_SIZE = 64

datagen = ImageDataGenerator(rescale=1. / 255)
train_data_dir = './images/normal_train/'
validation_data_dir = './images/normal_test/'
anomaly_data_dir = './images/anomaly/'
test_data_dir = './images/test'

train_generator = datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

anomaly_generator = datagen.flow_from_directory(
    directory=anomaly_data_dir,
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

test_generator = datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

# Subclassed Sequential Model
#conv_autoencoder = ImageAnomalyDetector()

# Model Training

# Functional Sequential Model
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                            input_shape=(SIZE, SIZE, 3), activity_regularizer=regularizers.l1(1e-6)))



conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
conv_autoencoder.add(Dropout(0.2))
conv_autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

conv_autoencoder.add(AttentionNetworkGen.AttentionBlock(32))

conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
conv_autoencoder.add(Dropout(0.2))
conv_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

conv_autoencoder.add(AttentionNetworkGen.AttentionBlock(16))

conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
conv_autoencoder.add(Dropout(0.2))

# Decoder
conv_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
conv_autoencoder.add(UpSampling2D((2, 2)))

conv_autoencoder.add(Dropout(0.2))
conv_autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

conv_autoencoder.add(AttentionNetworkGen.AttentionBlock(32))

conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Dropout(0.2))
conv_autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

conv_autoencoder.add(AttentionNetworkGen.AttentionBlock(64))

conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Dropout(0.2))
conv_autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
conv_autoencoder.compile(optimizer='adam', loss='mse')

conv_autoencoder.summary()


history = conv_autoencoder.fit(
    train_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=validation_generator,
    shuffle=True,
    callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
)


#visualkeras.layered_view(conv_autoencoder, to_file='cae_model.png') # write to disk

#KFold
"""
num_folds = 5
num_epochs = 50

train_data = np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())], axis=0)
results = []
# K-fold cross validation
fold_no = 0
for train_idx, test_idx in KFold(n_splits=num_folds, shuffle=True).split(train_data):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no + 1} ...')

    # Train/test splits
    x_train, x_test = train_data[train_idx], train_data[test_idx]

    # Modeli eğit
    history = conv_autoencoder.fit(x_train, x_train,
                              epochs=num_epochs,
                              batch_size=BATCH_SIZE,
                              validation_data=(x_test, x_test),
                              verbose=2)

    # Extract features from train and test images
    train_features = conv_autoencoder.predict(train_generator)
    test_features = conv_autoencoder.predict(test_generator)

    # Calculate reconstruction error
    train_mse = np.mean(np.power(x_train - train_features[:x_train.shape[0]], 2), axis=(1, 2, 3))
    test_mse = np.mean(np.power(x_test - test_features[:x_test.shape[0]], 2), axis=(1, 2, 3))

    # Calculate threshold using train set
    threshold = np.max(train_mse)

    # Evaluate performance on test set
    y_pred = (test_mse > threshold).astype(int)
    y_true = np.ones(len(x_test))
    y_true[len(train_data):] = 0

    # Append results
    results.append([accuracy_score(y_true, y_pred), f1_score(y_true, y_pred), precision_score(y_true, y_pred),
                    recall_score(y_true, y_pred)])

    # Print metrics
    print(f'Accuracy: {results[-1][0]}')
    print(f'F1-score: {results[-1][1]}')
    print(f'Precision: {results[-1][2]}')
    print(f'Recall: {results[-1][3]}')

    fold_no += 1

# Print average metrics over all folds
results = np.array(results)
print('Average metrics over all folds:')
print(f'Accuracy: {np.mean(results[:, 0])}')
print(f'F1-score: {np.mean(results[:, 1])}')
print(f'Precision: {np.mean(results[:, 2])}')
print(f'Recall: {np.mean(results[:, 3])}')

# Find best performing fold
best_fold = np.argmax(results[:, 1])
print(f'Best performing fold: {best_fold+1}')

conv_autoencoder.fit(train_generator, epochs=100)
score = conv_autoencoder.evaluate(test_generator)
print('Test Loss:', score)

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(scores)*100, np.std(scores)*100))
"""

# create_model func
"""
def create_model(dropout_rate = 0.1, l1_reg = 1e-6):
    conv_autoencoder = Sequential()
    conv_autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                                input_shape=(SIZE, SIZE, 3), activity_regularizer=regularizers.l1(l1_reg)))
    conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    conv_autoencoder.add(Dropout(dropout_rate))
    conv_autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    conv_autoencoder.add(Dropout(dropout_rate))
    conv_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    conv_autoencoder.add(Dropout(dropout_rate))

    # Decoder
    conv_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    conv_autoencoder.add(UpSampling2D((2, 2)))
    conv_autoencoder.add(Dropout(dropout_rate))
    conv_autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    conv_autoencoder.add(UpSampling2D((2, 2)))
    conv_autoencoder.add(Dropout(dropout_rate))
    conv_autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    conv_autoencoder.add(UpSampling2D((2, 2)))
    conv_autoencoder.add(Dropout(dropout_rate))
    conv_autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    conv_autoencoder.compile(optimizer='adam', loss='mae')
    return conv_autoencoder

"""


#conv_autoencoder.encoder.summary()
#conv_autoencoder.decoder.summary()

# Plot Loss

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot Accuracy
"""
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""

# Save Model
conv_autoencoder.save('./cae_model_new_2.h5', save_format='tf')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')

#conv_autoencoder = load_model('cae_model_new.h5')
conv_autoencoder.summary()


# Gridsearch
"""
# Create an instance of the KerasClassifier
conv_autoencoder = KerasClassifier(build_fn=create_model)

param_grid = {'l1_reg': [0.0001, 0.001, 0.01, 0.1],
              'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
              #'learning_rate': [0.0001, 0.001, 0.01, 0.1],
              #'lr_schedule': ['step', 'cosine'],
              #'loss': ['mse', 'mae']
              }

# TODO: Parametrelerde sorun var!
grid = GridSearchCV(estimator=conv_autoencoder,
                    param_grid=param_grid,
                    cv=5,
                    verbose=1,
                    scoring='accuracy',
                    n_jobs=-1)

grid_result = grid.fit(train_generator[0][0], train_label)
"""


# Plot reconstructed image sample
data_batch = []  # Capture all training batches as a numpy array
img_num = 0
while img_num <= train_generator.batch_index:  # gets each generated batch of size batch_size
    data = train_generator.next()
    data_batch.append(data[0])
    img_num = img_num + 1

predicted = conv_autoencoder.predict(data_batch[0])

image_number = random.randint(0, predicted.shape[0]-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(data_batch[0][image_number])
plt.title('Image ' + str(image_number) + ' Original')
plt.subplot(122)
plt.imshow(predicted[image_number])
plt.title('Image ' + str(image_number) + ' Reconstructed')
plt.show()


validation_error = conv_autoencoder.evaluate(validation_generator)
anomaly_error = conv_autoencoder.evaluate(anomaly_generator)

print("Recon. error for the validation (normal) data is: ", validation_error)
print("Recon. error for the anomaly data is: ", anomaly_error)

#for layer in conv_autoencoder.layers:
#    print(layer.name)

# Repeat prediction process for 10 times
for i in range(10):


    encoder_model = Sequential()
    encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3),
                             weights=conv_autoencoder.get_layer('conv2d').get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                             weights=conv_autoencoder.get_layer('conv2d_1').get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                             weights=conv_autoencoder.get_layer('conv2d_2').get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.summary()
    # encoder_model.input_shape --> (None, 32, 32, 3)

    # Get encoded output of input images = Latent space
    encoded_images = encoder_model.predict(train_generator)

    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape  # Here, we have 4x4x16
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]

    encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

    # Fit KDE to the image latent data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)

    # Calculate density and reconstruction error to find their means values for
    # good and anomaly images.
    # We use these mean and sigma to set thresholds.
    def calc_density_and_recon_error(batch_images):
        density_list = []
        recon_error_list = []
        for im in range(0, batch_images.shape[0] - 1):
            img = batch_images[im]
            img = img[np.newaxis, :, :, :]
            encoded_img = encoder_model.predict([[img]])  # Create a compressed version of the image using the encoder
            encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]  # Flatten the compressed image
            density = kde.score_samples(encoded_img)[0]  # get a density score for the new image
            reconstruction = conv_autoencoder.predict([[img]])
            reconstruction_error = conv_autoencoder.evaluate([reconstruction], [[img]], batch_size=1)
            density_list.append(density)
            recon_error_list.append(reconstruction_error)

        average_density = np.mean(np.array(density_list))
        stdev_density = np.std(np.array(density_list))

        average_recon_error = np.mean(np.array(recon_error_list))
        stdev_recon_error = np.std(np.array(recon_error_list))

        return average_density, stdev_density, average_recon_error, stdev_recon_error


    # Get average and std dev. of density and recon. error for normal and anomalous images.
    # For this let us generate a batch of images for each.
    train_batch = train_generator.next()[0]
    anomaly_batch = anomaly_generator.next()[0]

    normal_values = calc_density_and_recon_error(train_batch)
    anomaly_values = calc_density_and_recon_error(anomaly_batch)

    density_threshold = anomaly_values[0] + 2*anomaly_values[1]
    recon_threshold = anomaly_values[2] + 2*anomaly_values[3]

    normal_data_mean = np.mean(merged_data[:532])
    normal_data_stdev = np.std(merged_data[:532])
    anomaly_data_mean = np.mean(merged_data[532:])
    anomaly_data_stdev = np.std(merged_data[532:])


    def calc_chebyshev_thresholds(k_value):
        upper_chebyshev_th = normal_data_mean + k_value * anomaly_data_stdev
        lower_chebyshev_th = normal_data_mean - k_value * anomaly_data_stdev
        return lower_chebyshev_th, upper_chebyshev_th


    # P(∣X−μ∣≥kσ)≤ 1/k**2

    def check_chebyshev_inequlity(data_point, lower_chebyshev_th, upper_chebyshev_th):
        if np.any(data_point < lower_chebyshev_th) or np.any(data_point > upper_chebyshev_th):
            return True
        return False



    # Now, input unknown images and sort as Good or Anomaly
    def check_anomaly(img_path, density_threshold, recon_threshold, lower_chebyshev_th, upper_chebyshev_th):
        density_threshold = density_threshold  # Set this value based on the above exercise
        reconstruction_error_threshold = recon_threshold  # Set this value based on the above exercise
        predictions = np.array([])
        for image in img_path:
            img = Image.open(image)
            img = np.array(img.resize((32, 32), Image.LANCZOS))
            #plt.imshow(img)
            img = img / 255.
            img = img[np.newaxis, :, :, :3] # So as to make PIL image 3-channel, ":3" could be used.
            encoded_img = encoder_model.predict([[img]])
            encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
            reconstruction = conv_autoencoder.predict([[img]])

            density = kde.score_samples(encoded_img)[0]
            reconstruction_error = conv_autoencoder.evaluate([reconstruction], [[img]], batch_size=1)

            data_point = 0
            chebyshev_result = check_chebyshev_inequlity(data_point, lower_chebyshev_th, upper_chebyshev_th)
            data_point += 1

            # Anomaly Decision
            if reconstruction_error < reconstruction_error_threshold or density > density_threshold:
                print("The image is an anomaly")
                predictions = np.append(predictions, False)
            else:
                print("The image is NOT an anomaly")
                predictions = np.append(predictions, True)
        return predictions

    # Load test images and verify whether they are reported as anomalies.
    import glob

    preds = np.array([])

    test_path = glob.glob('images/test/test_images/*')

    k_values = np.linspace(0.1, 3, 1)

    rocs = []
    f1s = []
    # Display
    def display_stats(predictions, labels):
        print("Accuracy = {}".format(accuracy_score(labels, predictions)))
        print("Precision = {}".format(precision_score(labels, predictions)))
        print("Recall = {}".format(recall_score(labels, predictions)))
        print("F1-Score = {}".format(f1_score(labels, predictions)))

        f1s.append(f1_score(labels, predictions))

        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        rocs.append(roc_auc)
        print('ROC AUC Score = {}'.format(roc_auc))

        print(classification_report(labels, predictions, digits=3))

        LABELS = ["Anomaly", "Normal"]

        conf_matrix = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()

        plt.plot(fpr, tpr, linewidth=5, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], linewidth=5)
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # Test images verification
    for k_value in k_values:
        lower_chebyshev, upper_chebyshev = calc_chebyshev_thresholds(k_value)
        preds = np.append(preds, check_anomaly(test_path, density_threshold, recon_threshold, lower_chebyshev, upper_chebyshev))
        print(preds)

        preds = preds.astype(np.bool_)
        preds = tf.convert_to_tensor(preds)
        display_stats(preds, labels)
        preds = np.array([])

print(f"F1 Scores:{f1s}")
print(f"Average F1 Score: {sum(f1s) / len(f1s)}")
print(rocs)