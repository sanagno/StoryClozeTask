import os 
import sklearn 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm 
from sklearn.metrics import accuracy_score

# Custom dependencies 
from model import NLUModel
from utils import load_embedding
from data  import encode_text, shuffle_endings

LOG_PATH = "./log/lsdSem_lm/"
EMB_PATH = "./data/glove/glove.6B.100d.txt"

class WordBasedClassifier(NLUModel):
    """ Bidirectional LSTM Word-Based Classification Model """

    def __init__(self, vocab, sentence_len, drop_rate=0.5, batch_norm=False, embedding_dim=100, hidden_size=128):
        """
        Parameters:
        -----------
        vocab : dict
            The vocabulary dictionary. Keys correspond to unique words in the vocabulary and values correspond to the word ID. 

        sentence_len : int 
            The maximum length of a sentence. This should include '<bos>' and '<eos>' tokens. 

        drop_rate : float 
            The dropout rate, between 0 and 1. This is applied at the BiLSTM output and at the hidden layer of the feedforward network. 

        batch_norm : boolean 
            Whether or not to apply batch norm at the feedforward hidden layer. 

        embedding_dim : int 
            The dimensionality of the word embeddings. 

        hidden_size : int 
            The dimensionality of the hidden state. 
        """

        super(WordBasedClassifier, self).__init__('WordBasedClassifier')

        self.vocab = vocab 
        self.vocab_size = len(vocab)
        self.sentence_len = sentence_len
        self.context_len = 4*sentence_len 
        self.ending_len = sentence_len 
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Training indicator 
        self.training = tf.placeholder(dtype=tf.bool, shape=[], name="training_flag")

        self.__build()

    def __build(self):
        """
        Build the computational graph of the model 

        Returns:
        --------
        self : object 
            An instance of self. 
        """
        with tf.name_scope("Inputs"):
            self.context = tf.placeholder(dtype=tf.int32, shape=[None, self.context_len], name="input_context")
            self.ending  = tf.placeholder(dtype=tf.int32, shape=[None, self.ending_len] , name="input_ending")
            self.labels  = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")

        with tf.name_scope("EmbeddingLayer"):
            self.embeddings = tf.get_variable(name="embeddings", shape=[self.vocab_size, self.embedding_dim], dtype=tf.float32, 
                                    initializer=tf.initializers.random_uniform(-0.25, 0.25), trainable=True)

            # Embedding representation of context
            self.emb_context = tf.nn.embedding_lookup(self.embeddings, self.context) 

            # Embedding representation of ending 
            self.emb_ending  = tf.nn.embedding_lookup(self.embeddings, self.ending)


        with tf.variable_scope("ContextRNN"):
            # Forward Cell
            self.context_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                                        name="context_rnn_fw_cell")

            c_initial_fw_state = self.context_fw_cell.zero_state(batch_size=tf.shape(self.context)[0], dtype=tf.float32)

            # Backward Cell
            self.context_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                                        name="context_rnn_bw_cell")

            c_initial_bw_state = self.context_bw_cell.zero_state(batch_size=tf.shape(self.context)[0], dtype=tf.float32)

            # Unstack context tensor 
            c_inputs = tf.unstack(self.emb_context, axis=1)

            c_outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=self.context_fw_cell,
                                                             cell_bw=self.context_bw_cell, 
                                                             inputs=c_inputs, 
                                                             initial_state_fw=c_initial_fw_state, 
                                                             initial_state_bw=c_initial_bw_state, 
                                                             scope="context_rnn")

            c_stack_outputs = tf.stack(c_outputs, axis=1)

        with tf.variable_scope("EndingRNN"):
            # Forward Cell
            self.ending_fw_cell  = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                                        name="ending_rnn_fw_cell")

            e_initial_fw_state = self.ending_fw_cell.zero_state(batch_size=tf.shape(self.ending)[0], dtype=tf.float32)

            # Backward Cell
            self.ending_bw_cell  = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                                        name="ending_rnn_bw_cell")

            e_initial_bw_state = self.ending_bw_cell.zero_state(batch_size=tf.shape(self.ending)[0], dtype=tf.float32)

            # Unstack ending tensor 
            e_inputs = tf.unstack(self.emb_ending, axis=1)

            e_outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=self.ending_fw_cell,
                                                             cell_bw=self.ending_bw_cell, 
                                                             inputs=e_inputs, 
                                                             initial_state_fw=e_initial_fw_state, 
                                                             initial_state_bw=e_initial_bw_state, 
                                                             scope="ending_rnn")

            e_stack_outputs = tf.stack(e_outputs, axis=1)

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

        with tf.name_scope("IntermediateLayer"):
            feat = tf.concat([c_context_vec, e_context_vec], axis=1)

            # Apply dropout 
            feat = tf.layers.dropout(feat, rate=self.drop_rate, training=self.training)

        with tf.variable_scope("FeedForward"):
            h = tf.layers.dense(feat, units=64, activation=None, name="hidden_layer")

            if self.batch_norm: 
                h = tf.layers.batch_normalization(h, training=self.training)

            h = tf.nn.relu(h)

            h = tf.layers.dropout(h , rate=self.drop_rate, training=self.training)

            self.logits = tf.layers.dense(h, units=2, activation=None, name="output_layer")
            self.probs  = tf.nn.softmax(self.logits, name="softmax_activation")

        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits, name="loss"))

        return self

    def _optimizer(self, optimizer, learning_rate, clip_norm=10.0):
        """
        Define optimization method and training operation. The learning rate is decayed exponentially as training proceeds. 

        Parameters:
        -----------
        optimizer : string 
            Optimization method. Must be one of 'rms_prop', 'adam', 'adam_delta', 'sgd'.

        learning_rate : floar 
            The initial learning rate.

        clip_norm : float 
            The clipping ratio to be used for gradient clipping. 

        Returns: 
        --------
        self : object 
            An instance of self.
        """

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
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params, name="compute_gradients")

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm, name="clip_gradients") 

        with tf.name_scope("Minimize"):
            self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params),
                                                            global_step=self.global_step)
    
        return self

    def _batch_train(self, session, batch_x, batch_y, batch_x_val, batch_y_val, log_flag, print_flag):
        """
        Execute the training algorithm for a given batch. 

        Parameters:
        -----------
        session : tf.Session()
            A TensorFlow session. 

        batch_x : array-like, shape=(batch_size, 6*sentence_len)
            A batch of encoded training stories. Each row corresponds to a story. The first four sentences correspond to the context and the last two to the 
            two ending alternatives. 

        batch_y : array-like, shape=(batch_size, )
            A batch of training labels. Should be equal to either 1 or 2, indicating which ending alternative is the right one. 

        batch_x_val : array-like, shape=(batch_size, 6*sentence_len)
            A batch of encoded validation stories. Must be of the same structure as batch_x. 

        batch_y_val : array-like, shape=(batch_size, )
            A batch of validation labels. Should be equal to either 1 or 2, indicating which ending alternative is the right one.

        log_flag : boolean
            Whether to log summaries in TensorBoard. 

        print_flag : boolean
            Whether to print results to standard output.  

        Returns:
        --------
        self : object 
            An instance of self
        """
        batch_c = batch_x[:, :4*self.sentence_len] # Context
        batch_e = batch_x[:, 4*self.sentence_len:] # Endings 

        ##################   Ending 1   ##################
        batch_e_1 = batch_e[:, :self.sentence_len]

        # Labels for ending 1
        batch_y_1 = np.zeros(len(batch_y), dtype=np.int32)
        batch_y_1[batch_y==1] = 1

        train_dict = {self.context: batch_c, self.ending: batch_e_1, 
                        self.labels: batch_y_1, self.training: True}

        if log_flag:
            _, g_step, loss, summary = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
            self.train_loss.append(loss)
            self.train_writer.add_summary(summary, g_step)

            # Get validation batch 
            batch_c_val = batch_x_val[:, :4*self.sentence_len] # Context 
            batch_e_val = batch_x_val[:, 4*self.sentence_len:] # Endings

            # Ending 1
            batch_e_val_1 = batch_e_val[:, :self.sentence_len]

            # Labels 
            batch_y_val_1 = np.zeros(len(batch_y_val), dtype=np.int32)
            batch_y_val_1[batch_y_val==1] = 1

            valid_dict = {self.context: batch_c_val, self.ending: batch_e_val_1, 
                            self.labels: batch_y_val_1, self.training: False}

            summary = session.run(self.summaries, feed_dict=valid_dict)
            self.valid_writer.add_summary(summary, g_step)
        else:
            _, g_step, loss, _ = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
            self.train_loss.append(loss)

        ##################   Ending 2   ##################
        batch_e_2 = batch_e[:, self.sentence_len:]

        # Labels for ending 2
        batch_y_2 = np.zeros(len(batch_y), dtype=np.int32)
        batch_y_2[batch_y==2] = 1

        train_dict = {self.context: batch_c, self.ending: batch_e_2, 
                        self.labels: batch_y_2, self.training: True}

        if log_flag:
            _, g_step, loss, summary = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
            self.train_loss.append(loss)
            self.train_writer.add_summary(summary, g_step)

            # Get validation batch 
            batch_c_val = batch_x_val[:, :4*self.sentence_len] # Context 
            batch_e_val = batch_x_val[:, 4*self.sentence_len:] # Endings

            # Ending 1
            batch_e_val_2 = batch_e_val[:, self.sentence_len:]

            # Labels 
            batch_y_val_2 = np.zeros(len(batch_y_val), dtype=np.int32)
            batch_y_val_2[batch_y_val==2] = 1

            valid_dict = {self.context: batch_c_val, self.ending: batch_e_val_2, 
                            self.labels: batch_y_val_2, self.training: False}

            summary = session.run(self.summaries, feed_dict=valid_dict)
            self.valid_writer.add_summary(summary, g_step)
        else:
            _, g_step, loss, _ = session.run([self.train_step, self.global_step, self.loss, self.summaries], feed_dict=train_dict)
            self.train_loss.append(loss)

        # Report validation accuracy 
        if print_flag:
            batch_y_val_pred, _ = self._batch_predictions(session, batch_x_val)
            print('Validation Batch Accuracy: ', accuracy_score(batch_y_val, batch_y_val_pred))

        return self

    def fit(self, X, y, X_val, y_val, epochs=30, optimizer='adam', learning_rate=1e-4, batch_size=64, shuffle=True, log_freq=100, print_freq=1000):
        """
        Train the model on the provided data. 

        Parameters:
        -----------
        X : array-like, shape=(n_train, 6*sentence_len)
            Encoded training stories. Each row corresponds to a story. The first four sentences correspond to the context and the last two to the 
            two ending alternatives. 

        y : array-like, shape=(n_train, )
            Training labels. Should be equal to either 1 or 2, indicating which ending alternative is the right one. 

        X_val : array-like, shape=(n_train, 6*sentence_len)
            Encoded validation stories. Must be of the same structure as X. 

        y_val : array-like, shape=(n_train, )
            Validation labels. Must be of the same structure as y. 

        epochs : int 
            The epochs for which to train the model. 

        optimizer : string 
            The optimization method to be used. Must be one of 'rms_prop', 'adam', 'adam_delta', 'sgd'.

        learning_rate : float 
            The initial learning rate. 

        batch_size : int
            The batch size to be used. 

        shuffle : boolean 
            Whether to shuffle the training data at each epoch. 

        log_freq : int 
            The training step frequency at which to log summaries to TensorBoard. 

        print_freq : int
            The training step frequency at which to print results to standard output.  

        Returns:
        --------
        self : object 
            An instance of self.
        """

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

            # Load Glove embeddings 
            load_embedding(sess, self.vocab, self.embeddings, EMB_PATH, self.embedding_dim)

            n_train = X.shape[0]
            n_valid = X_val.shape[0]

            train_steps = n_train // batch_size
            valid_steps = n_valid // batch_size

            j_val = -1
            self.train_loss = []

            for epoch in tqdm(range(epochs), desc='Training'):
                print('Epoch %i' %(epoch+1))

                if shuffle: 
                    X, y = sklearn.utils.shuffle(X, y)

                for i in range(train_steps):

                    # Logging flags 
                    if i % log_freq == 0:
                        log_flag = True
                        # Validation logging 
                        j_val += 1
                        j = j_val % valid_steps
                    else:
                        log_flag = False

                    if i % print_freq == 0:
                        print_flag = True
                    else:
                        print_flag = False

                    # Training data
                    batch_x = X[i*batch_size:(i+1)*batch_size, :]
                    batch_y = y[i*batch_size:(i+1)*batch_size] 

                    # Validation data
                    batch_x_val = X_val[j*batch_size:(j+1)*batch_size, :]
                    batch_y_val = y_val[j*batch_size:(j+1)*batch_size] 

                    self._batch_train(sess, batch_x, batch_y, batch_x_val, batch_y_val, log_flag, print_flag)

                # Train using the remaining data
                batch_x = X[train_steps*batch_size:, :]
                batch_y = y[train_steps*batch_size:] 

                self._batch_train(sess, batch_x, batch_y, batch_x_val, batch_y_val, True, True)

                # Report validation accuracy every epoch 
                y_val_pred = np.array([])

                for i in range(valid_steps): 

                    batch_x_val = X_val[i*batch_size:(i+1)*batch_size, :]

                    y_pred_batch, _ = self._batch_predictions(sess, batch_x_val)

                    y_val_pred = np.append(y_val_pred, y_pred_batch) # numpy append 

                # Handle the remaining validation data 
                batch_x_val = X_val[valid_steps*batch_size:, :]
                y_pred_batch, _ = self._batch_predictions(sess, batch_x_val)

                y_val_pred = np.append(y_val_pred, y_pred_batch) # numpy append 

                print('Accuracy score at epoch %i : %.4f' %(epoch+1, accuracy_score(y_val, y_val_pred)))

                # Save the model at each epoch 
                saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=(epoch+1))  

        return self

    def predict(self, X, batch_size=64):
        """
        Perform classification on stories in X. 

        Parameters: 
        -----------
        X : array-like, shape=(n_samples, 6*sentence_len)
            Encoded stories. The first four sentences correspond to the story context and the last two to the two ending alternatives. 

        batch_size : int 
            Batch size to be used. 

        Returns: 
        --------
        y_pred : array-like, shape=(n_samples, )
            The predicted labels. Values are either 1 or 2 indicating which ending candidate is predicted as being true. 

        y_pred_proba : array-like, shape=(n_samples, 2)
            Predicted probabilities. For each sample and i = {1,2}, column i holds the probability of ending i being true. 
        """

        saver = tf.train.Saver()

        n_samples = X.shape[0]
        n_steps = n_samples // batch_size

        y_pred = np.array([])
        y_pred_proba = []

        with tf.Session() as sess:
            # Restore trained model
            saver.restore(sess,tf.train.latest_checkpoint(LOG_PATH))

            for i in range(n_steps): 

                batch_x = X[i*batch_size:(i+1)*batch_size, :]

                y_pred_batch, y_pred_proba_batch = self._batch_predictions(sess, batch_x)

                y_pred = np.append(y_pred, y_pred_batch) # numpy append
                y_pred_proba.append(y_pred_proba_batch) #list append

            # Handle the remaining points
            batch_x = X[n_steps*batch_size: , :]

            y_pred_batch, y_pred_proba_batch = self._batch_predictions(sess, batch_x)

            y_pred = np.append(y_pred, y_pred_batch) # numpy append
            y_pred_proba.append(y_pred_proba_batch) #list append

        y_pred_proba = np.concatenate(y_pred_proba, axis=0)

        return y_pred, y_pred_proba 

    def _batch_predictions(self, session, batch_x):
        """
        Perform classification for a single batch of stories. 
    
        Parameters: 
        -----------
        session : tf.Session()
            A TensorFlow session. 
        
        batch_x : array-like, shape=(batch_size, 6*sentence_len)
            Encoded stories. The first four sentences correspond to the story context and the last two to the two ending alternatives.

        Returns: 
        --------
        y_pred : array-like, shape=(batch_size, )
            The predicted labels. Values are either 1 or 2 indicating which ending candidate is predicted as being true. 

        y_pred_proba : array-like, shape=(batch_size, 2)
            Predicted probabilities. For each sample and i = {1,2}, column i holds the probability of ending i being true. 
        """

        batch_c = batch_x[:, :4*self.sentence_len] # Context
        batch_e = batch_x[:, 4*self.sentence_len:] # Endings 

        ##################   Ending 1   ##################
        batch_e_1 = batch_e[:, :self.sentence_len]

        valid_dict = {self.context: batch_c, self.ending: batch_e_1, self.training: False}
        
        y_pred_proba_e_1_ = session.run(self.probs, feed_dict=valid_dict)
        y_pred_proba_e_1  = y_pred_proba_e_1_[:,1] # Prob of e1 being true 

        ##################   Ending 2   ##################
        batch_e_2 = batch_e[:, self.sentence_len:]

        valid_dict = {self.context: batch_c, self.ending: batch_e_2, self.training: False}

        y_pred_proba_e_2_ = session.run(self.probs, feed_dict=valid_dict)
        y_pred_proba_e_2  = y_pred_proba_e_2_[:,1] # Prob of e1 being true 

        # Combine 
        y_pred_proba = np.hstack((y_pred_proba_e_1[:, np.newaxis], y_pred_proba_e_2[:, np.newaxis]))

        argmax = np.argmax(y_pred_proba, axis=1)
        y_pred = argmax + 1

        return y_pred, y_pred_proba

    def score(self, X, y, batch_size=64):
        """
        Returns the mean accuracy on the given test data and labels 

        Parameters: 
        -----------
        X : array-like, shape=(n_samples, 6*sentence_len)
            Encoded stories. The first four sentences correspond to the story context and the last two to the two ending alternatives. 

        batch_size : int 
            Batch size to be used. 

        Returns: 
        --------
        score : float   
            The mean accuracy of self.predict(X) with respect to y. 
        """

        y_pred, _ = self.predict(X, batch_size)
        score = accuracy_score(y, y_pred)

        return score

    def get_train_data(self, corpus, labels, shuffle=True):
        """
        Encode training corpus. 

        Parameters: 
        -----------
        corpus : array-like, shape=(n_train, 6)
            Training corpus. In each row, the first four columns correspond to the story context and the last two to the two ending alternatives. 

        labels : array-like, shape=(n_train, )
            Training labels. They should be either 1 or 2 indicating which ending alternative is correct. 

        shuffle : boolean 
            Whether to shuffle the ending alternatives. 

        Returns: 
        --------
        encoded_corpus : array-like, shape=(n_train, 6*sentence_len)
            Encoded training corpus (after shuffling). 

        shuffled_labels : array-like, shape=(n_train, )
            The training labels after shuffling. 
        """
        # Shuffle endings 
        if shuffle: 
            shuffled_corpus, shuffled_labels = shuffle_endings(corpus, labels) 

        encoded_context, encoded_endings = encode_text(shuffled_corpus, self.sentence_len, self.vocab)

        encoded_corpus = np.hstack((encoded_context, encoded_endings))
        
        return encoded_corpus, shuffled_labels

    def get_test_data(self, corpus): 
        """
        Encode testing corpus. 

        Parameters: 
        -----------
        corpus : array-like, shape=(n_train, 6)
            Testing corpus. In each row, the first four columns correspond to the story context and the last two to the two ending alternatives. 

        Returns: 
        --------
        encoded_corpus : array-like, shape=(n_train, 6*sentence_len)
            Encoded testing corpus.
        """
        encoded_context, encoded_endings = encode_text(corpus, self.sentence_len, self.vocab)

        encoded_corpus = np.hstack((encoded_context, encoded_endings))
        
        return encoded_corpus
