import numpy as np
import pandas as pd
import os
import copy
import librosa
import matplotlib.pyplot as plt
from keras.layers import LSTM, Flatten, Dropout, Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras


def find_max_and_return_index(mylist):
    co = 0
    mymax = 0
    for i,value in enumerate(mylist):
        if mymax < value:
            mymax = value
            co = i
    return mymax,co
list_folder = os.listdir('/home/nimnim/gunshot/gunshot2023summer/datasetsplited/train/')

hop_length = 512 #the default spacing between frames
n_fft = 255 #number of samples

raw_x = []
raw_y = []
for i in list_folder:
    list_file_wave = os.listdir('/home/nimnim/gunshot/gunshot2023summer/datasetsplited/train/' + i)
    for j in list_file_wave:
        filename = '/home/nimnim/gunshot/gunshot2023summer/datasetsplited/train/' + i + '/' + j
        data, sr = librosa.load(filename,sr=22050)

        raw_x.append(data)
        raw_y.append(i)
x = []
y = []
print("1")
for i in range(len(raw_x)):
    data = raw_x[i]
    if len(data) == 44100:
        mfcc_data = np.array(librosa.feature.mfcc(data, n_fft=n_fft,hop_length=hop_length,n_mfcc=128))
        x.append(mfcc_data)
        y.append(raw_y[i])

x_copy = copy.copy(x)
y_copy = copy.copy(y)


le = preprocessing.LabelEncoder()
le.fit(y_copy)
y_copy_new = le.transform(y_copy)
len_x_copy = []
for i in range(len(x_copy)):
    temp = x_copy[i].shape
    # print(temp)
    len_x_copy.append(temp)
value, count = np.unique(len_x_copy, return_counts=True)
print(value,count)
x_copy = np.array(x_copy)
X_train, X_test, y_train, y_test = train_test_split(x_copy, y_copy_new, test_size=0.25, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
input_shape = x_copy[0].shape

model = keras.Sequential()
model.add(LSTM(128,input_shape=input_shape,return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(9, activation='softmax'))
model.summary()
model.compile(loss="SparseCategoricalCrossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_val, y_val), shuffle=False)

model.save("gunshot_model.h5")
