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

    model.add(Conv2D(32, [3, 3], input_shape=(128, 1291, 1), padding='same'))
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
    model.add(Dense(2048, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # Add output neurons here (for now 4 calsses but will increase to 8 for medium and large data)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    return model

# Training the model
def training_model(model, X_train, Y_train, NUM_EPOCHS, BATCH_SIZE, X_VALIDATION, Y_VALIDATION):
    history = model.fit(X_train, Y_train,
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_VALIDATION, Y_VALIDATION),
              verbose=2)

    return model, history

# Predicting probabilities
def testing_model(model, X):
    return model.predict(X)

# Using pickle reading data
with open('Small/x_train_sm.pkl', 'rb') as f: X_train = pickle.load(f)
with open('Small/y_train_sm.pkl', 'rb') as f: Y_train = pickle.load(f)
with open('Small/x_test_sm.pkl', 'rb') as f: X_test = pickle.load(f)
with open('Small/y_test_sm.pkl', 'rb') as f: Y_test = pickle.load(f)
with open('Small/x_val_sm.pkl', 'rb') as f: X_validation = pickle.load(f)
with open('Small/y_val_sm.pkl', 'rb') as f: Y_validation = pickle.load(f)


# One-hot encoding
#encode the class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train_labels = encoder.transform(Y_train)
Y_train = to_categorical(Y_train_labels)
Y_test_labels = encoder.transform(Y_test)
Y_test = to_categorical(Y_test_labels)
Y_validation = to_categorical(encoder.transform(Y_validation))


# Getting data ready (Vaishnavi's code)
x_train = np.empty((len(X_train), len(X_train[0]), len(X_train[0][0])))
x_val = np.empty((len(X_validation), len(X_train[0]), len(X_train[0][0])))
x_test = np.empty((len(X_test), len(X_train[0]), len(X_train[0][0])))
for row in range(len(X_validation)): x_val[row] = np.array(X_validation[row])[:, :len(X_train[0][0])]
for row in range(len(X_test)): x_test[row] = np.array(X_test[row])[:, :len(X_train[0][0])]

for row in range(len(X_train)):
    if len(X_train[row][0]) < len(X_train[0][0]): x_train[row], x_train[row, :X_train[row].shape[0], :X_train[row].shape[1]] = np.zeros((len(X_train[0]), len(X_train[0][0]))), X_train[row]
    else: x_train[row] = np.array(X_train[row])[:, :len(X_train[0][0])]

X_train = x_train; X_validation = x_val; X_test = x_test

X_train = X_train.reshape(X_train.shape[0], 128, 1291, 1)
X_validation = X_validation.reshape(X_validation.shape[0], 128, 1291, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 1291, 1)

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
feature = open("x_train_sm_genre_cnn.csv", "wb")
np.save(feature, np.asarray(X_train_transform))
feature = open("x_test_sm_genre_cnn.csv", "wb")
np.save(feature, np.asarray(X_test_transform))

# Saving the train and validation accuracy as mat file
sio.savemat('small.mat', history.history)
