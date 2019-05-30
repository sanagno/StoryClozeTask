#!/bin/env python


import tensorflow as tf
import pandas as pd
import numpy as np
import csv, os
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Dropout, Lambda, GRU, Softmax
from keras.layers import Flatten, Concatenate, TimeDistributed, Layer
from keras.layers import Embedding
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
import keras.backend as KB
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import accuracy_score
from model import NLUModel

ENCODINGS_TRAIN = '../encodings.npy'
ENCODINGS_VAL = '../encodings_val.npy'
LABELS = '../labels.npy'

class ContextLSTM(NLUModel):

    def __init__(self, lstm_units=4800):
        super(ContextLSTM, self).__init__("ContextLSTM")
        self.lstm_units = lstm_units
        self.__build()

    def __build(self):
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dropout(0.1))
        softmax_layer = self.model.add(Dense(4800, activation='softmax'))

        def cosine_similarity_loss(layer):
            # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
            def loss(y_true,y_pred):
                pred_norm = y_pred/KB.batch_dot(y_pred,y_pred, axes=[1,1])
                true_norm = y_true/KB.batch_dot(y_true,y_true, axes=[1,1])
                cosine = KB.batch_dot(pred_norm, true_norm, axes=[1,1])
                cosine = KB.sum(cosine)
                return -cosine
            # Return a function
            return loss

        self.model.compile(loss=cosine_similarity_loss(softmax_layer),
                      optimizer='adam')

    def fit(self, X, y, epochs=10, batch_size=32):

        self.model.fit(X, y, epochs=epochs,
                  batch_size=batch_size)

    def predict(self, X):
        val_beg = X[0]
        val_end_1 = X[1]
        val_end_2 = X[2]
        predictions = self.model.predict(val_beg, batch_size=1)
        correct = 0
        final_predictions = np.zeros((predictions.shape[0],))
        for i in range(predictions.shape[0]):
            if cosine_similarity(predictions[i].reshape(1,-1), val_end_1[i].reshape(1,-1)) \
             < cosine_similarity(predictions[i].reshape(1,-1), val_end_2[i].reshape(1,-1)):
                pred = 1
            else:
                pred = 0

            final_predictions[i] = pred

        return final_predictions

    def get_train_data(self):
        train = np.load(ENCODINGS_TRAIN)

        train_beg = train[:,:4,:]
        train_end = train[:,4,:]
        return train_beg, train_end


    def get_test_data(self):
        val = np.load(ENCODINGS_VAL)
        val_beg = val[:,:4,:]
        val_end_1 = val[:,4,:]
        val_end_2 = val[:,5,:]
        y = np.load(LABELS)
        return [val_beg, val_end_1, val_end_2], y

