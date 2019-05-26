#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:47:37 2019

@author: yannis
"""

import pandas as pd
import numpy as np

# from keras_transformer.attention import MultiHeadAttention 
#   from skip_thoughts import configuration, encoder_manager
from keras.utils import to_categorical

MAX_NUM_WORDS = 200000
TRAINING_ENCODINGS = 'encodings_all.npy'
VALIDATION_ENCODINGS = 'encodings_val.npy'

#%% Load preprocessed data

train = np.load(TRAINING_ENCODINGS)

train_beg = train[:,:4,:]
train_end = train[:,4,:]

val = np.load(VALIDATION_ENCODINGS)
val_beg = val[:,:4,:]
val_end_1 = val[:,4,:]
val_end_2 = val[:,5,:]

y = np.load('labels.npy')
#y = to_categorical(y)

#%%
from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

fourth_beg_val = val[:,3,:]

cosine_similiarities = np.zeros((fourth_beg_val.shape[0]))

# Get the similarity between fourth and the wrong sentence
for i in range(fourth_beg_val.shape[0]):
    if y[i] == 0:
        wrong_end = val_end_2[i]
    else:
        wrong_end = val_end_1[i]
        
    cosine_similiarities[i] = cosine(fourth_beg_val[i], wrong_end)
    
# Learn Density of the similarty of fourth and the wrong one.
kde = KernelDensity(bandwidth=0.1)

kde.fit(cosine_similiarities.reshape(-1,1))

training = train_beg
rows = train_beg.shape[0]

end_cor = train_end
end_wrong = np.zeros((rows,4800))

end_wrong_i = np.zeros((rows,))

for i in range(rows):
# For every story sample a similarity between the fourth and the wrong sentence.
    sample = kde.sample()
    embedding = training[i,3]
    cosine_similarities_i = np.ones((rows,))
# Find the fifth sentence (which is not the correct fifth sentence of the story) which 
# is a similar to the fourth sentence as the sampled similarity
    for j in range(rows):
        if i == j: continue
        cosine_similarities_i[j] = abs(cosine(embedding, train_end[j]) - sample)
    
# Save the index that corresponds to the fifth sentence wrong sentence
    end_wrong_i[i] = np.argmin(cosine_similarities_i)
    
# Now we have essentially bootstrapped a training set like the validation set 
# For a story i, end_wrong_i[i] corresponds to another story, of which the fifth sentence we
# we are going to use as the wrong ending of story
np.save('wrong_endings.npy', end_wrong_i)
