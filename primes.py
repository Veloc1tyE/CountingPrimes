# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:53:42 2019

@author: billj
"""

import math
import numpy as np
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers

isPrime = [0,0,1,1,0,1,0,1]
primes = [2,3,5,7]
for i in range(8, 100001):
    flag = True
    for j in range(2, int(math.sqrt(i))+1):
        if i % j == 0:
           flag = False
    if flag:
        isPrime.append(1)
        primes.append(i)
    else:
        isPrime.append(0)
        
def prepare_data(isPrime, seq_length, n_points):
    seq_length = seq_length
    dataX = []
    dataY = []
    for i in range(0, n_points - seq_length, 1):
        seq_in = isPrime[i:i + seq_length]
        seq_out = isPrime[i + seq_length]
        dataX.append([data for data in seq_in])
        dataY.append(seq_out)
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    y = np_utils.to_categorical(dataY)
    return dataX, dataY, X, y

def build(X, y, optimizer, pretrained = False, filename = None):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(60))
    model.add(Dense(y.shape[1], activation='softmax'))
    if pretrained:
        model.load_weights(filename)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

if __name__ == '__main__':
    seq_length, n_points = 100, len(isPrime)
    dataX, dataY, X, y = prepare_data(isPrime, seq_length, n_points)
    adam = optimizers.adam(lr=1e-3, decay=1e-6)
    model = build(X, y, optimizer=adam)
    #filepath = "weights-improvement-" not too important atm
    #checkpoint
    model.fit(X, y, epochs=20, batch_size=128)
    