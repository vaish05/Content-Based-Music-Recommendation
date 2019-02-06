#importing necessary packages
import pandas as pd
import os
import pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np
home = os.getcwd()

#input to decide feature extraction for small or medium
os.chdir(os.path.join(home, 'fma_metadata'))
features = pd.read_csv('tracks.csv', header=[1])
inp = input('Data for small or medium [fma_small, fma_medium] ?-')


#extracting MFCC [128, 1291] with duration = 29.95, sampling rate=44100, hop_length=1024, signal window=2048
#categorizing the data to test, validation and training using the tracks.csv from fma
y_train, y_test, y_val, x_train, x_test, x_val, fil_train, fil_test, fil_val = [], [], [], [], [], [], [], [], []
genres = ['Blues', 'Hip-Hop', 'Pop', 'Classical', 'Jazz', 'Rock', 'Country', 'Folk']
folders = [folder for folder in os.listdir(os.chdir(os.path.join(home, inp))) if os.path.isdir(folder)]
val_cnt, tst_cnt, trn_cnt, cnt = 0, 0, 0, 1
for dire in folders:
    files = [file for file in os.listdir(os.chdir(os.path.join(home, inp+"\\"+dire))) if '.mp3' in file]
    for fil in files:
        if features['track_id'].astype(str).str.contains(fil.replace('.mp3', '').lstrip('0')).any():
            file_genre = features.loc[features['track_id'] == int(fil.replace('.mp3', '')), 'genre_top'].to_string(index = False)
            file_split = features.loc[features['track_id'] == int(fil.replace('.mp3', '')), 'split'].to_string(index = False)
            if file_genre in genres:
                x, sr = librosa.load(fil, duration = 29.95, sr=44100)
                if x.shape[0] != 1320795: continue
                log_mel_spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, sr=sr, n_mels=128, hop_length=1024, n_fft=2048))
                if file_split == 'training' or (file_split == 'test' and tst_cnt >= 100):
                    x_train.append(log_mel_spec)
                    y_train.append(file_genre)
                    fil_train.append(fil)
                elif file_split == 'validation':
                    x_val.append(log_mel_spec)
                    y_val.append(file_genre)
                    fil_val.append(fil)
                elif file_split == 'test':
                    x_test.append(log_mel_spec)
                    y_test.append(file_genre)
                    fil_test.append(fil)
                    tst_cnt = tst_cnt+1


# Checking the length of the data extracted
print(len(x_test), len(y_test), len(fil_test))
print(len(x_train), len(y_train), len(fil_train))
print(len(x_val), len(y_val), len(fil_val))

# Specify small or medium dataset for file
ext = input('Data for small or medium [sm, md] ?-')
os.chdir(os.path.join(home, 'features'))

# Dump song file names for playlist generation
with open('file_train_' + ext + '.pkl', 'wb') as f: pickle.dump(fil_train, f, protocol=2)
with open('file_test_' + ext + '.pkl', 'wb') as f: pickle.dump(fil_test, f, protocol=2)
with open('file_val_' + ext + '.pkl', 'wb') as f: pickle.dump(fil_val, f, protocol=2)

# Dumping validation and testing data to files
with open('y_test_' + ext + '.pkl', 'wb') as f: pickle.dump(y_test, f, protocol=2)
with open('x_test_' + ext + '.pkl', 'wb') as f: pickle.dump(x_test, f, protocol=2)
with open('y_val_' + ext + '.pkl', 'wb') as f: pickle.dump(y_val, f, protocol=2)
with open('x_val_' + ext + '.pkl', 'wb') as f: pickle.dump(x_val, f, protocol=2)
with open('y_train_' + ext + '.pkl', 'wb') as f: pickle.dump(y_train, f, protocol=2)

# The features information for training dataset is pretty large to dump from memory, so writing data in splits of 2500 records
cnt = 0
for i in range(0, len(x_train), 2500):
    print('writing', i, '-', i+2500)
    temp, cnt = x_train[i:i+2500], cnt+1
    with open('x_train_' + ext + '_' + str(cnt) + '.pkl', 'wb') as f: pickle.dump(temp, f, protocol=2)
    del temp

# Clearing memory
del x_train
del y_train
del y_test
del y_val
del x_test
del x_val
del fil_test
del fil_train
del fil_val
del f

#Extracing label information to understand data distribution
ext = input('Data for small or medium [sm, md] ?-')
os.chdir(os.path.join(home, 'features'))
with open('y_test_' + ext + '.pkl', 'rb') as f: test = pickle.load(f)
with open('y_val_' + ext + '.pkl', 'rb') as f: val = pickle.load(f)
with open('y_train_' + ext + '.pkl', 'rb') as f: train = pickle.load(f)

# Displayig data distribution in test, training and validation data and in small/medium datasets
from collections import Counter
plt.figure()
label_count = Counter(test)
xtr = pd.DataFrame.from_dict(label_count, orient = 'index')
xtr.plot(kind = 'bar')
plt.title('Test')

plt.figure()
label_count = Counter(train)
xtr = pd.DataFrame.from_dict(label_count, orient = 'index')
xtr.plot(kind = 'bar')
plt.title('Train')

plt.figure()
label_count = Counter(val)
xtr = pd.DataFrame.from_dict(label_count, orient = 'index')
xtr.plot(kind = 'bar')
plt.title('Validation')

labels = train
labels.extend(val)
labels.extend(test)
plt.figure()
label_count = Counter(labels)
xtr = pd.DataFrame.from_dict(label_count, orient = 'index')
xtr.plot(kind = 'bar')
plt.title('All')