# Creator : Akshay Naik (aunaik)

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from keras.models import Sequential, Model
from sklearn.utils import class_weight
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from keras import applications
import scipy.io as sio


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CNN model architecture
def Create_network():
    model = Sequential()

    model.add(Conv2D(32, [3, 3], input_shape=(128, 1290, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(64, [3, 3], activation='relu', input_shape=(128, 1291, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4,4)))

    model.add(Conv2D(64, [3, 3], padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(128, [3, 3], activation='relu', input_shape=(128, 1291, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    model.add(Conv2D(128, [3, 3], activation='relu', padding='same'))#, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, [3, 3], padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    #model.add(Conv2D(256, [3, 3], activation='relu', input_shape=(128, 1291, 1), padding='same'))
    #model.add(Conv2D(256, [3, 3], activation='relu', input_shape=(128, 1291, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    #model.add(Dropout(0.2))

    # Flattening the data
    model.add(Flatten())

    # Adding Fully connected layers
    # model.add(Dropout(0.5))

    #model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.7))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))  # Add output neurons here (for now 4 calsses but will increase to 8 for medium and large data)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-04),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    return model


# Training the model
def training_model(model, X_train, Y_train, NUM_EPOCHS, BATCH_SIZE, X_VALIDATION, Y_VALIDATION):
    history = model.fit(X_train, Y_train,
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_VALIDATION, Y_VALIDATION),
	      class_weight=class_weights,
              verbose=2)

    return model, history

# Predicting probabilities
def testing_model(model, X):
    return model.predict(X)

# Reading data
# Merging train data for medium dataset
for i in range(1,6):
    print(i)
    if i == 1: X_train = np.array(pickle.load(open('Medium/x_train_md_'+str(i)+'.pkl', 'rb')))
    else: X_train = np.append(X_train, np.array(pickle.load(open('Medium/x_train_md_'+str(i)+'.pkl', 'rb'))), axis = 0)

with open('Medium/y_train_md.pkl', 'rb') as f: Y_train = np.array(pickle.load(f))
with open('Medium/x_test_md.pkl', 'rb') as f: X_test = np.array(pickle.load(f))
with open('Medium/y_test_md.pkl', 'rb') as f: Y_test = np.array(pickle.load(f))
with open('Medium/x_val_md.pkl', 'rb') as f: X_validation = np.array(pickle.load(f))
with open('Medium/y_val_md.pkl', 'rb') as f: Y_validation = np.array(pickle.load(f))


# Calculating class weights
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)))


# One-hot encoding
#encode the class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train_labels = encoder.transform(Y_train)
Y_train = to_categorical(Y_train_labels)
Y_test_labels = encoder.transform(Y_test)
Y_test = to_categorical(Y_test_labels)
Y_validation = to_categorical(encoder.transform(Y_validation))



X_train = X_train.reshape(X_train.shape[0], 128, 1290, 1)
X_validation = X_validation.reshape(X_validation.shape[0], 128, 1290, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 1290, 1)

print (X_train.shape, X_test.shape, X_validation.shape)

# Initializing parameters
NUM_EPOCHS = 100
BATCH_SIZE = 8

# Creating network
model = Create_network()


# Training network
model, history = training_model(model, X_train, Y_train, NUM_EPOCHS, BATCH_SIZE, X_validation, Y_validation)


# Transforming signal data to genre space
X_train_transform = testing_model(model, X_train)
X_test_transform = testing_model(model, X_test)

# Calculating traina dn test accuracies
print('Train Accuracy: {}%'.format(100 * accuracy_score(Y_train_labels, np.argmax(X_train_transform, axis=1))))
print('Test Accuracy: {}%'.format(100 * accuracy_score(Y_test_labels, np.argmax(X_test_transform, axis=1))))


# Saving the transformed data
feature = open("x_train_md_genre_cnn1.csv", "wb")
np.save(feature, np.asarray(X_train_transform))
feature = open("x_test_md_genre_cnn1.csv", "wb")
np.save(feature, np.asarray(X_test_transform))

# Saving the train and validation accuracy as mat file
sio.savemat('medium1.mat', history.history)
