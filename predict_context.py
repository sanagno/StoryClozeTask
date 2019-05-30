#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:53:34 2019

@author: yannis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:47:37 2019

@author: yannis
"""

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



# from keras_transformer.attention import MultiHeadAttention 
#   from skip_thoughts import configuration, encoder_manager
from keras.utils import to_categorical
VAL_SET = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
ROC_VAL_SET = 'ROCStories__spring2016 - ROCStories_spring2016.csv'
TEST_SET = 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
DATA_DIR = 'dataset'
ENCODER_PATH = 'finetune-transformer-lm-master/model/encoder_bpe_40000.json'
BPE_PATH = 'finetune-transformer-lm-master/model/vocab_40000.bpe'
EMB_PATH = 'wordembeddings-dim100.word2vec'

MAX_NUM_WORDS = 200000

#%% Load preprocessed data
train = np.load('encodings.npy')

train_beg = train[:,:4,:]
train_end = train[:,4,:]

val = np.load('encodings_val.npy')
val_beg = val[:,:4,:]
val_end_1 = val[:,4,:]
val_end_2 = val[:,5,:]

y = np.load('labels.npy')
#y = to_categorical(y)

#%% Train model only on ROC KDE sampling

training = train_beg
rows = train_beg.shape[0]

end_cor = train_end

labels = np.concatenate([np.zeros((rows,)), np.ones((rows,))], axis=0)
labels = to_categorical(labels)

perm = np.random.permutation(2*rows)

#%% Define model

model = Sequential()
model.add(LSTM(128))
model.add(Dropout(0.1))
softmax_layer = model.add(Dense(4800, activation='softmax'))

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

model.compile(loss=cosine_similarity_loss(softmax_layer),
              optimizer='adam')

#%% Run and evaluate

model.fit(train_beg, train_end, validation_split=0.3, epochs=10, 
          batch_size=16)

#%% Define evaluation model

predictions = model.predict(val_beg, batch_size=1)
correct = 0
for i in range(predictions.shape[0]):
    if cosine_similarity(predictions[i].reshape(1,-1), val_end_1[i].reshape(1,-1)) \
     < cosine_similarity(predictions[i].reshape(1,-1), val_end_2[i].reshape(1,-1)):
        pred = 1
    else:
        pred = 0
    if pred == y[i]:
        correct+=1

print(correct/ predictions.shape[0])