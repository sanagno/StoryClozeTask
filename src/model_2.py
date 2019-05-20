import os 
import sklearn 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm 
from sklearn.metrics import accuracy_score

class RNNBaseline(object): 

    def __init__(self, 
                 cell, 
                 input_size, 
                 seq_len=5, 
                 hidden_size=1000):
        """
        Initialize model parameters and define the computational graph. 

        Parameters:
        -----------
        cell : string 
            Cell type to be used. Current cell types are: 'basic', 'lstm', 'gru'.

        input_size : int
            The dimensionality of the input (at each time step).

        dropout_rate : float 
            The dropout rate. Applied at the feedforward classifier. 

        seq_len : int
            Specifies the length of of each sequence to be fed in the RNN. 

        hidden_size : int
            The dimensionality of the hidden state in the RNN. 

        Returns: 
        --------
        self : object 
            An instance of self
        """

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        with tf.name_scope("inputs"):
            # Embedded stories of shape 
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.input_size], name="embedded_stories")

            # Labels 
            self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")

        with tf.name_scope("rnn"):
            if cell=='basic':
                self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size, 
                                                    name="rnn_cell")
            elif cell=='lstm':
                self.cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, 
                                                    name="rnn_cell")
            elif cell=='gru':
                self.cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, 
                                                    name="rnn_cell")
            else:
                raise ValueError('Invalid cell type provided')

            initial_state = self.cell.zero_state(batch_size=tf.shape(self.x)[0], dtype=tf.float32)

            # Unstack x to get a length T list of inputs, each a tensor of shape [batch_size, input_size]
            inputs = tf.unstack(self.x, axis=1)

            outputs, state = tf.nn.static_rnn(cell=self.cell, 
                                              inputs=inputs, 
                                              initial_state=initial_state,
                                              scope="static_rnn")
            
        with tf.variable_scope("feedforward"):
            self.logits = tf.layers.dense(inputs=outputs[-1], units=2, activation=None, name="output_layer")

        with tf.name_scope("probs"):
            self.probs = tf.nn.softmax(self.logits, name="softmax_activation")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="loss"))

    def optimizer(self, optimizer='rms_prop', learning_rate=0.001, clip_norm=10.0):
        """
        Compute, clip and gradients 

        Parameters:
        -----------
        optimizer : string 
            Type of optimizer. 

        clip_norm : A 0-D (scalar) Tensor > 0
            The clipping ratio.

        Returns:
        --------
        self : object 
            An instance of self
        """
        with tf.name_scope("learning_rate"):
            self.global_step = tf.Variable(0, trainable=False)
            decay_steps = 2000
            decay_rate = 0.96

            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name="learning_rate")
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)

        with tf.name_scope("optimizer"):
            # Optimizer
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

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params, name="compute_gradients")

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm, name="clip_gradients")

        with tf.name_scope("apply_gradients"):
            self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params), global_step=self.global_step)

        return self

    def train(self, x, y, x_val, y_val, epochs=10, optimizer='rms_prop', learning_rate=0.001, batch_size=128, shuffle=True, log_freq=50):
        """
        Trains the model for a given number of epochs (iterations on a dataset).

        Parameters:
        -----------
        x : array-like, shape=(n_train, story_len, input_size)
            Embedded training stories. 

        y : array-like, shape=(n_train, )
            Training story labels. 

        x_val : array-like, shape=(n_val, story_len, input_size)
            Embedded validation stories. 

        y_val : array-like, shape=(n_val, )
            Validation story labels.

        epochs : int 
            Number of epochs to train the model. An epoch is an iteration over the entire x data provided.

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 128.
        
        shuffle : Boolean
            Whether to shuffle the training data before each epoch

        log_freq : int
            Frequency of logging results to summaries 
        
        Returns:
        --------
        self : object 
            An instance of self
        """

        LOG_PATH = "./log/rnn_baseline/"

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
        tf.summary.scalar('training_loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        merged_summaries = tf.summary.merge_all()

        with tf.Session(config=config) as sess:
            # Summaries 
            train_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'training'), sess.graph)

            # Run the initializer
            sess.run(init)

            N_train = x.shape[0]
            N_valid = x_val.shape[0]

            # Floor division 
            train_steps = N_train // batch_size
            valid_steps = N_valid // batch_size

            # Logging  
            j_val = 0
            train_loss = []

            for epoch in range(epochs):
                print('Epoch %i' %(epoch+1))

                if shuffle: 
                    x, y = sklearn.utils.shuffle(x, y)

                for i in range(train_steps):

                    # Get training batch 
                    batch_x = x[i*batch_size:(i+1)*batch_size, :, :]
                    batch_y = y[i*batch_size:(i+1)*batch_size]

                    train_dict = {self.x: batch_x, self.y: batch_y}

                    if i % log_freq == 0:
                        # Training step 
                        _, g_step, loss, summary = sess.run([self.train_step, self.global_step, self.loss, merged_summaries], feed_dict=train_dict)
                        train_loss.append(loss)
                        train_writer.add_summary(summary, g_step)

                        # Use modulo indexing 
                        j = j_val % valid_steps

                        # Get validation batch 
                        batch_x_val = x_val[j*batch_size:(j+1)*batch_size, :, :]
                        batch_y_val = y_val[j*batch_size:(j+1)*batch_size]

                        # Get batch prediction 
                        y_val_pred, _ = self.batch_predictions(sess, batch_x_val)

                        # Increment the couter 
                        j_val += 1

                    else:
                        _, g_step, loss, summary = sess.run([self.train_step, self.global_step, self.loss, merged_summaries], feed_dict=train_dict)
                        train_loss.append(loss)
                        train_writer.add_summary(summary, g_step)

                # Train on the remaining stories 
                batch_x = x[train_steps*batch_size:, :, :]
                batch_y = y[train_steps*batch_size:]

                train_dict = {self.x: batch_x, self.y: batch_y}

                _, g_step, loss, summary = sess.run([self.train_step, self.global_step, self.loss, merged_summaries], feed_dict=train_dict)
                train_loss.append(loss)
                train_writer.add_summary(summary, g_step)

                # Save the model at each epoch 
                saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=(epoch+1))  

        plt.figure()
        plt.plot(np.array(train_loss))
        plt.ylabel('Training Loss')
        plt.xlabel('Training Steps')
        plt.grid(True)
        plt.savefig('train_loss.eps', format='eps', dpi=1000)

        return self

    def predict(self, x, batch_size=128):
        """
        Generates output predictions for the input stories. 

        Parameters:
        -----------
        x : array-like, shape=(n_stories, story_len)
            The input stories (embedded).

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 100.
        
        Returns:
        --------
        y_pred : array-like, shape=(n_stories, )
            Array of predicted labels.

        y_pred_proba : array-like, shape=(n_stories, )
            Compute probabilities of possible outcomes for samples in x.
        """
        LOG_PATH = "./log/rnn_baseline/"

        saver = tf.train.Saver()

        # Batch processing 
        N = x.shape[0]
        n_steps = N // batch_size

        # Initialization 
        y_pred = np.array([])
        y_pred_proba = []

        with tf.Session() as sess:
            # Restore trained model
            saver.restore(sess,tf.train.latest_checkpoint(LOG_PATH))

            for i in range(n_steps):

                batch_x = x[i*batch_size:(i+1)*batch_size, :, :]

                # Compute predicted labels for current batch 
                y_pred_, y_pred_proba_ = self.batch_predictions(sess, batch_x)

                y_pred = np.append(y_pred, y_pred_) # numpy append 
                y_pred_proba.append(y_pred_proba_) # list append 

            # Handle the remaining data points 
            batch_x = x[n_steps*batch_size:, :, :]

            y_pred_, y_pred_proba_ = self.batch_predictions(sess, batch_x)

            y_pred = np.append(y_pred, y_pred_) # numpy append 
            y_pred_proba.append(y_pred_proba_) # list append 

        y_pred_proba = np.concatenate(y_pred_proba, axis=0)

        return y_pred, y_pred_proba

    def batch_predictions(self, session, batch_x):
        # Ending 1
        mask_1 = [True, True, True, True, True, False]

        batch_x_1 = batch_x[:, mask_1, :]

        y_pred_proba_e1_ = session.run(self.probs, feed_dict={self.x: batch_x_1})
        y_pred_proba_e1 = y_pred_proba_e1_[:,1]

        # Ending 1
        mask_2 = [True, True, True, True, False, True]
        
        batch_x_2 = batch_x[:, mask_2, :]

        y_pred_proba_e2_ = session.run(self.probs, feed_dict={self.x: batch_x_2})
        y_pred_proba_e2 = y_pred_proba_e2_[:,1]

        # Combine 
        y_pred_proba = np.hstack((y_pred_proba_e1[:, np.newaxis], y_pred_proba_e2[:, np.newaxis]))

        argmax = np.argmax(y_pred_proba, axis=1)
        y_pred = argmax + 1

        return y_pred, y_pred_proba

    def score(self, x, y, batch_size=128):
        """
        Returns the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        x : array-like, shape=(n_stories, story_len)
            The input stories (embedded).

        y : array-like, shape=(n_stories, )

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 100.
        
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(x) w.r.t. y.
        """

        y_pred, _ = self.predict(x, batch_size)

        score = accuracy_score(y, y_pred)

        return score
