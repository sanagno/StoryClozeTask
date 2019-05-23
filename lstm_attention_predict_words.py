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
from keras.layers import Dense, LSTM, Input, Dropout, Lambda
from keras.layers import Flatten, Concatenate, TimeDistributed
from keras.layers import Embedding
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
import keras.backend as KB
from sklearn.model_selection import train_test_split
from keras_transformer.attention import MultiHeadAttention 

VAL_SET = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
ROC_VAL_SET = 'ROCStories__spring2016 - ROCStories_spring2016.csv'
TEST_SET = 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
DATA_DIR = 'dataset'
ENCODER_PATH = 'finetune-transformer-lm-master/model/encoder_bpe_40000.json'
BPE_PATH = 'finetune-transformer-lm-master/model/vocab_40000.bpe'
EMB_PATH = 'wordembeddings-dim100.word2vec'

MAX_NUM_WORDS = 200000

#%% Retrieve Data
roc = pd.read_csv(DATA_DIR + "/" + ROC_VAL_SET, nrows = 40000)\
    .drop(columns=['storyid','storytitle'])

sentences = ['sentence%d'%i for i in range(1,5)]

beginnings = roc.apply(lambda x:  ' '.join(x[sentences]), axis=1)
endings = roc['sentence5']

corpus_roc = roc.apply(lambda x: ' '.join(x))


val = pd.read_csv(DATA_DIR + "/" + VAL_SET, nrows=10000).drop(columns=['InputStoryid'])

sentences = ['InputSentence%d'%i for i in range(1,5)]

corpus_val = val.drop(columns='AnswerRightEnding').apply(lambda x: ' '.join(x))
beginnings_val = val.apply(lambda x:  ' '.join(x[sentences]), axis=1)

corpus = pd.concat([corpus_roc, corpus_val])

#%% Tokenize
#tokenizer = Tokenizer(num_words = MAX_NUM_WORDS,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, oov_token='<unk>')
tokenizer.fit_on_texts(corpus)
input_seq = tokenizer.texts_to_sequences(beginnings)
output_seq = tokenizer.texts_to_sequences(endings)

special_tokens = dict()
special_tokens['<bos>'] = len(tokenizer.word_index)
special_tokens['<eos>'] = len(tokenizer.word_index) + 1
special_tokens['<pad>'] = 0

input_seq_val = tokenizer.texts_to_sequences(beginnings_val)
output_seq_val_1 = tokenizer.texts_to_sequences(val['RandomFifthSentenceQuiz1'])
output_seq_val_2 = tokenizer.texts_to_sequences(val['RandomFifthSentenceQuiz2'])

def add_bos_eos(seq):
    return [special_tokens['<bos>']] + seq + [special_tokens['<eos>']]


output_seq = list(map(add_bos_eos, output_seq))
output_seq_val_1 = list(map(add_bos_eos,output_seq_val_1))
output_seq_val_2 = list(map(add_bos_eos,output_seq_val_2))


input_max_len = max([max(len(a),len(b)) for a,b in zip(input_seq, input_seq_val)])
output_max_len = max([max(len(a),len(b),len(c)) for a,b,c in zip(output_seq, output_seq_val_1, output_seq_val_2)])

vocab_size = len(tokenizer.word_index) + 1 + 2 # 2 for eos and bos

X_train = pad_sequences(input_seq, padding='post', maxlen=input_max_len)
#y_output_seq = pad_sequences(output_seq, padding='post', maxlen=output_max_len+1)
#y_input_seq = np.roll(y_output_seq, 1)
y_input_seq = pad_sequences(output_seq, padding='post', maxlen=output_max_len+1)
y_output_seq = np.roll(y_input_seq, -1)
y_output_seq[:,-1] = 0


input_val = pad_sequences(input_seq_val, padding='post', maxlen=input_max_len)
output_val_1 = pad_sequences(output_seq_val_1, padding='post', maxlen=output_max_len+1)
output_val_2 = pad_sequences(output_seq_val_2, padding='post', maxlen=output_max_len+1)

#%% Load embeddings
# load the whole embedding into memory
embeddings_index = dict()
f = open(EMB_PATH)
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

#%% Create embedding matrix
nones = 0
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        nones +=1 

print("%d unknown words"%nones)

#%% Define model

hidden_layer = 128

encoder_input = Input((X_train.shape[1],), name="encoder_input")
eel = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=True, name="embedding_encoder")
ee = eel(encoder_input)
encoder_out, encoder_state_h, encoder_state_c = LSTM(hidden_layer, return_sequences=True, return_state=True, name="encoder_lstm")(ee)
#encoder_out = Lambda(lambda x: KB.reshape(x, (-1, 21, 128)))(encoder_out)

decoder_input = Input((y_input_seq.shape[1],), name="decoder_input")
ed = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=True, name="embedding_decoder")
decoder_embedding = ed(decoder_input)

decoder_lstm = LSTM(hidden_layer, return_sequences=True, return_state=True, name="decoder_lstm")

decoder_out, _, _ = decoder_lstm(decoder_embedding, initial_state=[encoder_state_h, encoder_state_c])
#decoder_out = Lambda(lambda x: KB.reshape(x, (-1, y_input_seq.shape[1], 128)))(decoder_out)

attention_layer = MultiHeadAttention(32, True)
attn_out = attention_layer([encoder_out, decoder_out])

attn_out = Lambda(lambda x: KB.reshape(x, (-1, y_input_seq.shape[1], 128)))(attn_out)
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])
decoder_concat_input = Lambda(lambda x: KB.reshape(x, (-1, y_input_seq.shape[1], 256)))(decoder_concat_input)

#print(decoder_concat_input)
#print(decoder_concat_input)
dense_layer = Dense(vocab_size, activation='softmax', name="softmax")

dense_time_layer = TimeDistributed(dense_layer, name='time_distributed_layer')
dense_time = dense_time_layer(decoder_concat_input)

#softmax = dense_layer(decoder_out)

def loss_fun(layer):
    return sparse_categorical_crossentropy

def generate_data(X_train, y_input_seq, y_output_seq, vocab_size, batch_size=16):
    while True:
        for b in range(0, X_train.shape[0]//batch_size):
            y_output = np.zeros((batch_size,y_output_seq.shape[1], vocab_size))
            for i in range(batch_size):
                for j in range(y_output_seq.shape[1]):
                    y_output[i,j, y_output_seq[i][j]] = 1
            yield [X_train[b*batch_size:(b+1)*batch_size], y_input_seq[b*batch_size: (b+1)*batch_size]], \
                    y_output_seq[b*batch_size:(b+1)*batch_size]

def normal_loss(y_true, y_pred):
    shape = y_pred.get_shape()
    print(shape)
    y_true = KB.reshape(y_true,(-1,shape[-2]))
    y_true = KB.cast(y_true,'int32')
    mask = KB.cast(KB.not_equal(y_true, 0), 'float32')
    n = KB.sum(mask, axis=-1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)*mask
    loss = KB.sum(loss, axis=-1)
    loss = loss / n
    return(tf.reduce_mean(loss))

optimizer = Adam(clipnorm=5.)
model = Model([encoder_input, decoder_input], dense_time)
model.compile(loss=normal_loss, optimizer=optimizer)
#model.fit([X_train, y_input_seq], y_output_seq, batch_size=16, epochs=5)
#model.fit_generator(generate_data(X_train, y_input_seq, y_output_seq, vocab_size), 
#                    steps_per_epoch = X_train.shape[0]//16, epochs=1)
model.fit([X_train,y_input_seq],y_output_seq, batch_size=64, epochs=3)
model.save('s2s.h5')

#%% Inference Model

encoder_model = Model(encoder_input, [encoder_out, encoder_state_h, encoder_state_c])


previous_word = Input(shape=(1,), name='probs')
dec_embed = ed(previous_word)


encoder_output_input = Input(shape=(X_train.shape[1], hidden_layer))
decoder_state_input_h = Input(shape=(hidden_layer,))
decoder_state_input_c = Input(shape=(hidden_layer,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(dec_embed, initial_state=decoder_states_inputs)

attn_out_inf = attention_layer([encoder_output_input, decoder_outputs])
#attn_out_inf = Lambda(lambda x: KB.reshape(x, (-1, y_input_seq.shape[1], 128)), name="first_lambda")(attn_out_inf)
attn_out_inf = Lambda(lambda x: KB.reshape(x[:,0,:], (-1, 1, 128)))(attn_out_inf)


decoder_states = [state_h, state_c]

inf_concat = Concatenate(axis=-1)([decoder_outputs, attn_out_inf])

dense_outputs = dense_layer(inf_concat)

decoder_model = Model(
    [previous_word] + [encoder_output_input] + decoder_states_inputs,
    [dense_outputs] + decoder_states) 

inference_model = {'encoder':encoder_model, 'decoder':decoder_model}
#%% Predict continuations

def predict_continuation(sentence, inference_model, maxlen,endchar):
    sentence = sentence.reshape(1,-1)
    encoder_out, state_h, state_c = inference_model['encoder'].predict(sentence)
    states = [state_h, state_c]
    outputs = []
    word_id = np.zeros((1,))
    for i in range(maxlen):
        output, state_h, state_c = inference_model['decoder'].predict([word_id, encoder_out] + states)
        output = output.reshape(1,-1)
        outputs.append(np.argmax(output))
        word_id[0] = np.argmax(output, axis=1)
        if word_id[0] == endchar:
            break
        states = [state_h, state_c]
        
    return outputs
#%%

outputs = np.zeros((output_val_1.shape[0], output_val_1.shape[1]))
for i in range(output_val_1.shape[0]):
    output = predict_continuation(input_val[i], inference_model,output_val_1.shape[1], special_tokens['<eos>'])
    outputs[i,:len(output)] = output
    

#%% Final Model 
y = val['AnswerRightEnding'] - 1

y = y.values
        
ff_input = Input(shape=(3, output_val_1.shape[1]))
ff_embed = eel(ff_input)
ff_flatten = Flatten()(ff_embed)
ff_dense = Dense(64, activation='relu')(ff_flatten)
ff_activation = Dense(1, activation='sigmoid')(ff_dense)

ff_model = Model(ff_input, ff_activation)

ff_model.compile(loss="binary_crossentropy", optimizer='adam',
	metrics=["accuracy"])

input_to_ff = np.array([outputs, output_val_1, output_val_2])
input_to_ff = np.swapaxes(input_to_ff, 0,1 )
print(input_to_ff.shape)
ff_model.fit(input_to_ff, y, batch_size=16, epochs=20, validation_split=0.3)
