# Import necessary packages
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
import scipy.io as sio
from keras.layers import Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle


# Default lstm setting for training the network
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.

opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-08)

batch_size = 64
nb_epochs = 10

#Data extraction of training, testing and validation - labels and features
home = os.getcwd()
features = "features"
ext = input('Enter file type [sm, md]:')

os.chdir(os.path.join(home, features))
with open('y_test_'+ ext +'.pkl', 'rb') as f: Y_test = np.array(pickle.load(f))
with open('y_val_'+ ext +'.pkl', 'rb') as f: Y_validation = np.array(pickle.load(f))
with open('y_train_'+ ext +'.pkl', 'rb') as f: Y_train = np.array(pickle.load(f))

with open('x_test_'+ ext +'.pkl', 'rb') as f: X_test = np.array(pickle.load(f))
with open('x_val_'+ ext +'.pkl', 'rb') as f: X_validation = np.array(pickle.load(f))

with open('file_test_'+ ext +'.pkl', 'rb') as f: F_test = np.array(pickle.load(f))
with open('file_val_'+ ext +'.pkl', 'rb') as f: F_validation = np.array(pickle.load(f))
with open('file_train_'+ ext +'.pkl', 'rb') as f: F_train = np.array(pickle.load(f))

for i in range(1, 6):
    if i == 1: X_train = np.array(pickle.load(open('x_train_' + ext + '_' + str(i) + '.pkl', 'rb')))
    else: X_train = np.append(X_train, np.array(pickle.load(open('x_train_' + ext + '_' + str(i) + '.pkl', 'rb'))), axis = 0)


# Assign class weights for unbalaned dataset
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)))

# Generating one-hot encoding for label information to generate probability distribution in genre space
encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train_labels = encoder.transform(Y_train)
Y_test_labels = encoder.transform(Y_test)
Y_validation_labels = encoder.transform(Y_validation)
Y_train = to_categorical(Y_train_labels)
Y_test = to_categorical(Y_test_labels)
Y_validation = to_categorical(Y_validation_labels)

# Function to define network 
def create_network(Y_train):
    print('Building the LSTM-RNN model')
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=(len(X_train[0]), len(X_train[0][0]))))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=Y_train.shape[1], activation='softmax'))
    
    print("Compiling the model")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


# Training the model
def training_model(model, X_train, Y_train, nb_epochs, batch_size, X_validation, Y_validation):
    history = model.fit(np.asarray(X_train), Y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_validation, Y_validation), verbose=2)
                        #, class_weight=class_weights)
    return history, model


# Predicting probabilities
def testing_model(model, X):
    return model.predict(X)


# Creating network
model = create_network(Y_train)

# Training network
history, model = training_model(model, X_train, Y_train, nb_epochs, batch_size, X_validation, Y_validation)

# Transforming signal data to genre space
X_train_transform = testing_model(model, X_train)
X_test_transform = testing_model(model, X_test)

print('Train Accuracy: {}%'.format(accuracy_score(Y_train_labels, np.argmax(X_train_transform, axis=1))))
print('Test Accuracy: {}%'.format(accuracy_score(Y_test_labels, np.argmax(X_test_transform, axis=1))))

# Saving the model information
sio.savemat('model_' + ext +'.mat', history.history)

# Saving the transformed data
feature = open("x_train_genre.csv", "wb")
np.save(feature, np.asarray(X_train_transform))
feature = open("x_test_genre.csv", "wb")
np.save(feature, np.asarray(X_test_transform))

