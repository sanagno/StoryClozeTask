import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import sklearn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils import shuffle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import bert
from tensorflow.keras.layers import *
import sys
from tqdm import tqdm
import os
from load_embeddings import load_glove_model,load_glove_emb

class Glove_and_Sent2vec():
  
  def __init__(self, vocab, sentence_len, drop_rate=0.5, batch_norm=False, embedding_dim=100, hidden_size=128,hidden_size_sent2vec=256,sent2vec_size=500,tune_embds=True):

      self.vocab = vocab.copy()
      self.vocab_size = len(vocab)
      self.sentence_len = sentence_len
      self.context_len = 4*sentence_len
      self.ending_len = sentence_len
      self.drop_rate = drop_rate
      self.batch_norm = batch_norm
      self.embedding_dim = embedding_dim
      self.hidden_size = hidden_size
      self.tune_embds=tune_embds
      
      #sent2vec
      self.num_sentences=5
      self.sentence_size=sent2vec_size
      self.hidden_size_sent2vec=hidden_size_sent2vec

      # Training indicator 
      self.training = tf.placeholder(dtype=tf.bool, shape=[], name="training_flag")
      
      
      
      with tf.name_scope("Inputs"):
          self.context = tf.placeholder(dtype=tf.int32, shape=[None, self.context_len], name="input_context")
          self.ending  = tf.placeholder(dtype=tf.int32, shape=[None, self.ending_len] , name="input_ending")
          self.labels  = tf.placeholder(dtype=tf.float32, shape=[None,1], name="labels")
          
          #sent2vec
          self.doc=tf.placeholder(dtype=tf.float32,shape=[None,self.num_sentences,self.sentence_size],name='document')
     

      with tf.name_scope("EmbeddingLayer"):
          
          self.embedding_lookup_=Embedding(self.vocab_size, self.embedding_dim,trainable=self.tune_embds,mask_zero=True)

          self.emb_context = self.embedding_lookup_(self.context)
          
          self.emb_ending  = self.embedding_lookup_(self.ending)


      with tf.variable_scope("ContextRNN"):
          
          self.context_fw_cell=LSTM(self.hidden_size,return_sequences=True,kernel_initializer='he_normal')

          c_stack_outputs=Bidirectional(self.context_fw_cell)(self.emb_context)
  
          

      with tf.variable_scope("EndingRNN"):

          self.ending_fw_cell=LSTM(self.hidden_size,return_sequences=True,kernel_initializer='he_normal')
                    
          e_stack_outputs=Bidirectional(self.context_fw_cell)(self.emb_ending)
          

      with tf.variable_scope("ContextAttention"):
          c_score = tf.layers.dense(inputs=tf.layers.dense(c_stack_outputs, units=16, activation=tf.math.tanh), 
                              units=1, name="context_score")

          c_attention_weights = tf.nn.softmax(c_score, axis=1)

          c_context_vec = c_attention_weights * c_stack_outputs
          c_context_vec = tf.reduce_sum(c_context_vec, axis=1)

      with tf.variable_scope("EndingAttention"):
          e_score = tf.layers.dense(inputs=tf.layers.dense(e_stack_outputs, units=4, activation=tf.math.tanh), 
                              units=1, name="ending_score")

          e_attention_weights = tf.nn.softmax(e_score, axis=1)

          e_context_vec = e_attention_weights * e_stack_outputs
          e_context_vec = tf.reduce_sum(e_context_vec, axis=1)
      
      with tf.variable_scope("Sent2vec_BidLSTM"):
   
        lstm_cell=LSTM(self.hidden_size_sent2vec,kernel_initializer='he_normal')
        
        self.last_state=Bidirectional(lstm_cell,merge_mode='concat')(self.doc)
        
          
      with tf.name_scope("IntermediateLayer"):
        
          feat = tf.concat([c_context_vec, e_context_vec,self.last_state], axis=1) 
          
          feat = Dropout(self.drop_rate)(feat)

      with tf.variable_scope("FeedForward"):
             
        h = Dense(256, activation=None, name="hidden_layer")(feat)

        if self.batch_norm: 
            h = tf.layers.batch_normalization(h, training=self.training)

        h = tf.nn.relu(h)

        h = Dropout(self.drop_rate)(h)
        
        self.logits = Dense(1, activation=None, name="output_layer")(h)
        self.probs  = tf.nn.sigmoid(self.logits, name="sigmoid_activation")
     

      with tf.name_scope("Loss"):
          self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits, name="loss"))

  def optimizer(self, optimizer, learning_rate, clip_norm=10.0):

          with tf.name_scope("LearningRate"):
              decay_steps = 3000
              decay_rate = 0.97
              self.global_step = tf.Variable(0, trainable=False)

              self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name="learning_rate")
              self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)

          with tf.name_scope("Optimizer"):
              if optimizer=='rms_prop':
                  self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
              elif optimizer=='adam':
                  self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
              elif optimizer=='adadelta':
                  self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
              elif optimizer=='sgd':
                  self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
              else:
                  raise ValueError('Invalid optimization method provided')

          with tf.name_scope("GradientComputation"):
              
              gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
             
              clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm, name="clip_gradients") 

          with tf.name_scope("Minimize"):
              
              self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, variables),global_step=self.global_step)

          return self

  def batch_train(self, session, batch_x,batch_sent2vec_tr, batch_y, batch_x_val,batch_sent2vec_val, batch_y_val, log_flag):
      """
      Execute training update for a given batch. 

      Parameters:
      -----------

      Returns:
      --------
      self : object 
          An instance of self
      """
      batch_c = batch_x[:, :4*self.sentence_len] # Context
      batch_e = batch_x[:, 4*self.sentence_len:] # Endings 

      # Ending 1
      batch_e_1 = batch_e[:, :self.sentence_len]
      
      #sent2vec with ending 1
      sent2vec1=batch_sent2vec_tr[:,:5,:]

      #Labels for ending 1
      batch_y_1 = np.zeros(len(batch_y), dtype=np.int32)
      batch_y_1[batch_y==1] = 1   
  
      train_dict = {self.context: batch_c, self.ending: batch_e_1,\
                    self.labels: batch_y_1.reshape(-1,1),self.doc:sent2vec1, self.training: True}

      if log_flag:
          _, g_step, loss, summary = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
          self.train_loss.append(loss)
          self.train_writer.add_summary(summary, g_step)

          # Get validation batch 
          batch_c_val = batch_x_val[:, :4*self.sentence_len] # Context 
          batch_e_val = batch_x_val[:, 4*self.sentence_len:] # Endings

          # Ending 1
          batch_e_val_1 = batch_e_val[:, :self.sentence_len]
          
          #sent2vec 1
          sent2vec1_val=batch_sent2vec_val[:,:5,:]

          # Labels 
          batch_y_val_1 = np.zeros(len(batch_y_val), dtype=np.int32)
          batch_y_val_1[batch_y_val==1] = 1

          valid_dict = {self.context: batch_c_val, self.ending: batch_e_val_1,\
                        self.labels: batch_y_val_1.reshape(-1,1),self.doc:sent2vec1_val, self.training: False}

          summary = session.run(self.summaries, feed_dict=valid_dict)
          self.valid_writer.add_summary(summary, g_step)
      else:
          _, g_step, loss, _ = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
          self.train_loss.append(loss)

      # Ending 2
      batch_e_2 = batch_e[:, self.sentence_len:]
      
      #sent2vec with ending 2
      sent2vec2=batch_sent2vec_tr[:,[0,1,2,3,5],:]

      # Labels for ending 2
      batch_y_2 = np.zeros(len(batch_y), dtype=np.int32)
      batch_y_2[batch_y==2] = 1

      train_dict = {self.context: batch_c, self.ending: batch_e_2,\
                    self.labels: batch_y_2.reshape(-1,1),self.doc:sent2vec2, self.training: True}

      if log_flag:
          _, g_step, loss, summary = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
          self.train_loss.append(loss)
          self.train_writer.add_summary(summary, g_step)

          # Get validation batch 
          batch_c_val = batch_x_val[:, :4*self.sentence_len] # Context 
          batch_e_val = batch_x_val[:, 4*self.sentence_len:] # Endings

          # Ending 1
          batch_e_val_2 = batch_e_val[:, self.sentence_len:]
          
          #sent2vec 2
          sent2vec2_val=batch_sent2vec_val[:,[0,1,2,3,5],:]

          # Labels 
          batch_y_val_2 = np.zeros(len(batch_y_val), dtype=np.int32)
          batch_y_val_2[batch_y_val==2] = 1

          valid_dict = {self.context: batch_c_val, self.ending: batch_e_val_2,\
                        self.labels: batch_y_val_2.reshape(-1,1),self.doc:sent2vec2_val, self.training: False}

          summary = session.run(self.summaries, feed_dict=valid_dict)
          self.valid_writer.add_summary(summary, g_step)
      else:
          _, g_step, loss, _ = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
          self.train_loss.append(loss)

      return self

  def train(self, x_train,sent2vec_train, y_train, x_val,sent2vec_val, y_val,glove_path, epochs=30, optimizer='adam', learning_rate=1e-4, batch_size=64,shuffle=True, log_freq=100):

      LOG_PATH = "./log/lsdSem/"
      EMB_PATH = glove_path

      # GPU options 
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = "0"
      
      # Define Optimizer
      self.optimizer(optimizer, learning_rate)

      # Model saver
      saver = tf.train.Saver(tf.global_variables())

      # Initialize the variables 
      init = tf.global_variables_initializer()

      # Tensorboard 
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('learning_rate', self.learning_rate)
      self.summaries = tf.summary.merge_all()

      with tf.Session(config=config) as sess:
          # Summaries 
          self.train_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'training'), sess.graph)
          self.valid_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'validation'))

          # Run the initializer
          sess.run(init)

          weights=load_glove_emb(self.vocab,EMB_PATH,self.embedding_dim)
          self.embedding_lookup_.set_weights([weights])

          N_train = x_train.shape[0]
          N_valid = x_val.shape[0]

          train_steps = N_train // batch_size
          valid_steps = N_valid // batch_size

          j_val = -1
          self.train_loss = []

          for epoch in tqdm(range(epochs), desc='Training'):
              print('Epoch %i' %(epoch+1))

              if shuffle: 
                  x_train, y_train, sent2vec_train = sklearn.utils.shuffle(x_train, y_train, sent2vec_train)

              for i in range(train_steps):

                  if i % log_freq == 0:
                      log_flag = True
                      # Validation logging 
                      j_val += 1
                      j = j_val % valid_steps
                  else:
                      log_flag = False

                  # Training data
                  batch_x = x_train[i*batch_size:(i+1)*batch_size, :]
                  batch_sent2vec_tr=sent2vec_train[i*batch_size:(i+1)*batch_size, :]
                  batch_y = y_train[i*batch_size:(i+1)*batch_size] # note shape ([None])

                  # Validation data
                  batch_x_val = x_val[j*batch_size:(j+1)*batch_size, :]
                  batch_sent2vec_val=sent2vec_val[j*batch_size:(j+1)*batch_size, :]
                  batch_y_val = y_val[j*batch_size:(j+1)*batch_size] # note shape ([None])

                  self.batch_train(sess, batch_x,batch_sent2vec_tr, batch_y, batch_x_val,batch_sent2vec_val, batch_y_val, log_flag)

              # Train using the remaining data
              batch_x = x_train[train_steps*batch_size:, :]
              batch_y = y_train[train_steps*batch_size:] 
              
              batch_sent2vec_tr=sent2vec_train[train_steps*batch_size:, :]
              

              self.batch_train(sess, batch_x,batch_sent2vec_tr, batch_y, batch_x_val,batch_sent2vec_val, batch_y_val, False)

              # Save the model at each epoch 
              saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=(epoch+1))  

      return self

  def predict(self, x,sent2vec, batch_size=64):

      LOG_PATH = "./log/lsdSem/"

      saver = tf.train.Saver()

      N = x.shape[0]
      n_steps = N // batch_size

      y_pred = np.array([])
      y_pred_proba = []

      with tf.Session() as sess:
          # Restore trained model
          saver.restore(sess,tf.train.latest_checkpoint(LOG_PATH))

          for i in range(n_steps): 

              batch_x = x[i*batch_size:(i+1)*batch_size, :]
              
              sent2vec_batch=sent2vec[i*batch_size:(i+1)*batch_size, :]

              y_pred_batch, y_pred_proba_batch = self.batch_predictions(sess, batch_x,sent2vec_batch)

              y_pred = np.append(y_pred, y_pred_batch) # numpy append
              y_pred_proba.append(y_pred_proba_batch) #list append

          # Handle the remaining points
          batch_x = x[n_steps*batch_size: , :]
          
          sent2vec_batch=sent2vec[n_steps*batch_size: , :]
          
          y_pred_batch, y_pred_proba_batch = self.batch_predictions(sess, batch_x,sent2vec_batch)

          y_pred = np.append(y_pred, y_pred_batch) # numpy append
          y_pred_proba.append(y_pred_proba_batch) #list append

      y_pred_proba = np.concatenate(y_pred_proba, axis=0)

      return y_pred, y_pred_proba 

  def batch_predictions(self, session, batch_x,sent2vec_batch):

      batch_c = batch_x[:, :4*self.sentence_len] # Context
      batch_e = batch_x[:, 4*self.sentence_len:] # Endings 
      
      sent2vec1_val=sent2vec_batch[:,:5,:]

      # Ending 1
      batch_e_1 = batch_e[:, :self.sentence_len]

      valid_dict = {self.context: batch_c, self.ending: batch_e_1,self.doc: sent2vec1_val, self.training: False}

      y_pred_proba_e_1_ = session.run(self.probs, feed_dict=valid_dict)
      y_pred_proba_e_1  = y_pred_proba_e_1_[:] # Prob of e1 being true 

      # Ending 2
      batch_e_2 = batch_e[:, self.sentence_len:]
      
      sent2vec2_val=sent2vec_batch[:,[0,1,2,3,5],:]

      valid_dict = {self.context: batch_c, self.ending: batch_e_2,self.doc:sent2vec2_val, self.training: False}

      y_pred_proba_e_2_ = session.run(self.probs, feed_dict=valid_dict)
      y_pred_proba_e_2  = y_pred_proba_e_2_[:] # Prob of e1 being true 

      # Combine 
      y_pred_proba = np.hstack((y_pred_proba_e_1[:, np.newaxis], y_pred_proba_e_2[:, np.newaxis]))

      argmax = np.argmax(y_pred_proba, axis=1)
      y_pred = argmax + 1

      return y_pred, y_pred_proba

  def score(self, x, sent2vec, y, batch_size=64):

      y_pred, _ = self.predict(x,sent2vec, batch_size)
      score = accuracy_score(y, y_pred)

      return score
