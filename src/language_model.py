import os 
import sklearn 
import numpy as np 
import tensorflow as tf

from tqdm import tqdm 

# Custom dependencies
from data import decode_sentence
from utils import load_embedding

LOG_PATH = "./log/language_model/"
EMB_PATH = "./data/glove/glove.6B.100d.txt"

class LanguageModel(object):
    """ RNN-based Language Model with LSTM cell"""
    
    def __init__(self, vocab, inverse_vocab, sentence_len, embedding_dim=100, hidden_size=512):
        """
        Parameters: 
        -----------
        vocab : dict
            The vocabulary dictionary. Keys correspond to unique words in the vocabulary and values correspond to the word ID. 

        inverse_vocab : dict 
            The inverse of the vocabulary where keys correspond to (unique) IDs in the vocabulary 
            and the values correspond to the associated word. 

        sentence_len : int 
            The maximum length of a sentence. This should include '<bos>' and '<eos>' tokens. 

        embedding_dim : int 
            The dimensionality of the word embeddings. 

        hidden_size : int 
            The dimensionality of the hidden state. 
        """
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.vocab_size = len(vocab)
        self.sentence_len = sentence_len
        self.story_len = 5*sentence_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size 

        # Build computational graphs 
        self._create_main_graph()
        self._create_one_step_graph()

    def _create_main_graph(self):
        """
        Build the computational graph of the model. 

        Returns: 
        --------
        self : object 
            An instance of self. 
        """
        with tf.name_scope("Input"):
            self.story = tf.placeholder(dtype=tf.int32, shape=[None, self.story_len], name="input_story")

        with tf.name_scope("EmbeddingLayer"):
            self.embeddings = tf.get_variable(name="embeddings", shape=[self.vocab_size, self.embedding_dim], dtype=tf.float32, 
                                    initializer=tf.initializers.random_uniform(-0.25, 0.25), trainable=True)

            self.emb_story = tf.nn.embedding_lookup(self.embeddings, self.story)
        
        with tf.name_scope("ParameterInitialization"):
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                                initializer=tf.contrib.layers.xavier_initializer(), 
                                                name="rnn_cell")

            state = self.cell.zero_state(batch_size=tf.shape(self.story)[0], dtype=tf.float32)

        total_loss = []

        with tf.name_scope("StaticRNN"):
            # Unroll the RNN Loop 
            for t in range(self.story_len - 1):

                inputs = self.emb_story[:, t, :]

                labels = self.story[:, t+1]

                outputs, state = self.cell(inputs, state)

                logits = self._output_layer(outputs, output_dim=self.vocab_size, reuse=(t > 0))

                total_loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        with tf.name_scope("Loss"):
            # Bring to shape [batch_size, sentence_len - 1]
            total_loss = tf.transpose(total_loss)

            # Ignore '<pad>' symbols in loss computation 
            mask = tf.cast(tf.not_equal(self.story[:,1:], self.vocab['<pad>']), dtype=tf.float32)

            masked_loss = mask * total_loss

            # Add up losses in each sentence and divide by sentence len (excluding '<pad>' symbols)
            loss_per_sentence = tf.reduce_sum(masked_loss, axis=1) / tf.reduce_sum(mask, axis=1)

            # Total batch loss
            self.loss = tf.reduce_mean(loss_per_sentence)

        with tf.name_scope("Perplexity_Computation"):

            self.perplexity_per_sentence = tf.exp(loss_per_sentence)

            self.mean_perplexity = tf.reduce_mean(self.perplexity_per_sentence)

        return self

    def _create_one_step_graph(self):
        """
        Create a one-timestep graph for the model. 

        Returns:
        --------
        self : object 
            An instance of self. 
        """
        with tf.name_scope("OneStepGraph"):
            self.one_step_word_index = tf.placeholder(tf.int32, [1], name="one_step_input_token")
            self.one_step_state_1 = tf.placeholder(tf.float32, [1, self.hidden_size], name="hidden_state_1")
            self.one_step_state_2 = tf.placeholder(tf.float32, [1, self.hidden_size], name="hidden_state_2")

            one_step_word_emb = tf.nn.embedding_lookup(self.embeddings, self.one_step_word_index, name='one_step_word_emb')

            one_step_output, self.one_step_new_state = self.cell(one_step_word_emb, (self.one_step_state_1,
                                                                                     self.one_step_state_2))

            # Project onto the vocabulary 
            # logits has shape [1, vocab_size]
            logits = self._output_layer(one_step_output, output_dim=self.vocab_size, reuse=True)

            # Reshape into [vocab_size]
            # logits = tf.reshape(logits, [-1]) 

            self.one_step_next_word = tf.argmax(logits, axis=1)
            # self.one_step_next_word_probs = tf.nn.softmax(logits)

        return self

    def _output_layer(self, inputs, output_dim, reuse):
        """
        Output layer projecting onto vocabulary. 

        Parameters: 
        -----------
        inputs : Tensor
            Input data. 

        output_dim : int
            Dimensionality of the output space.

        reuse : Boolean 
            Whether to reuse the weights of a previous layer by the same name. 
        """
        with tf.variable_scope("OutputLayer", reuse=reuse):
            return tf.layers.dense(inputs, output_dim, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), name="output_layer")

    def _optimizer(self, optimizer, learning_rate, clip_norm=5.0):
        """
        Compute, clip and gradients 

        Parameters:
        -----------
        optimizer : string 
            Optimization method. Must be one of 'rms_prop', 'adam', 'adam_delta', 'sgd'.

        learning_rate : floar 
            The initial learning rate.

        clip_norm : A 0-D (scalar) Tensor > 0
            The clipping ratio to be used for gradient clipping. 

        Returns:
        --------
        self : object 
            An instance of self
        """
        with tf.name_scope("LearningRate"):
            self.global_step = tf.Variable(0, trainable=False)
            self.global_epoch = tf.Variable(0, trainable=False)

            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name="learning_rate")

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

    def _train_epoch(self, session, data, batch_size, shuffle, log_freq):
        """
        Perform one training epoch on the provided data.

        Parameters:
        -----------
        session : tf.Session() 
            A TensorFlow session. 

        data : array-like, shape=(n_train, story_len)
            Encoded stories. 

        batch_size : int 
            Batch size.
        
        shuffle : boolean 
            Whether to shuffle the training data at the end of each epoch. 

        log_freq : int
            The training step frequency at which to log summaries to TensorBoard. 

        Returns: 
        --------
        self : object 
            An instance of self
        """
        n_steps = data.shape[0] // batch_size

        if shuffle: 
            data = sklearn.utils.shuffle(data)

        for i in range(n_steps):
            # Get mini-batch 
            batch_x = data[i*batch_size:(i+1)*batch_size, :]

            if i % log_freq == 0: 
                _, g_step, summary = session.run([self.train_step, self.global_step, self.summaries], feed_dict={self.story: batch_x})
                self.train_writer.add_summary(summary, g_step)
            else:
                _, _, _ = session.run([self.train_step, self.global_step, self.summaries], feed_dict={self.story: batch_x})

        # Train using the remaining data 
        batch_x = data[n_steps*batch_size:, :]
        _, _, _ = session.run([self.train_step, self.global_step, self.summaries], feed_dict={self.story: batch_x})

        return self

    def train(self, train_data, val_finetune, training_blocks=3, training_epochs=2, tunning_epochs=5, optimizer='adam', learning_rate=1e-3, batch_size=64, shuffle=True, log_freq=50):
        """
        Train the model by alternating between the training set and the validation set. 

        Parameters:
        -----------
        train_data : array-like, shape=(n_train, story_len)
            Training stories encoded in terms of IDs in the vocabulary. 

        val_finetune : array-like, shape=(n_train, story_len)
            Training stories encoded in terms of IDs in the vocabulary. 

        training_blocks : int 
            Each training block consists of performing 'training_epochs' on the training data and 'tunning_epochs' on the finetunning data.

        training_epochs : int 
            The number of training epochs in each training block. 

        tunning_epochs : int 
            The number of tunning epochs in each training block. 

        optimizer : string 
            Optimization method to be used for training. 

        learning_rate : float 
            Starting learning rate. 

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 64.

        shuffle : Boolean
            Whether to shuffle the training data at each epoch.

        log_freq : int 
            The training step frequency at which to log summaries to TensorBoard. 

        Returns:
        --------
        self : object 
            An instance of self
        """

        # GPU options 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"

        # Define Optimizer
        self._optimizer(optimizer, learning_rate)

        # Saver
        saver = tf.train.Saver(tf.global_variables())

        # Initialize the variables 
        init = tf.global_variables_initializer()

        # Tensorboard 
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('perplexity', self.mean_perplexity)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.summaries = tf.summary.merge_all()

        with tf.Session(config=config) as sess:
            # Summaries 
            self.train_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'training'), sess.graph)

            # Run the initializer
            sess.run(init)

            # Load Glove embeddings 
            load_embedding(sess, self.vocab, self.embeddings, EMB_PATH, self.embedding_dim)

            for _ in tqdm(range(training_blocks), desc='Training Block'):

                for _ in tqdm(range(training_epochs), desc='Training'):

                    self._train_epoch(sess, train_data, batch_size, shuffle, log_freq)
                    self.global_epoch += 1

                    # Save the model at each epoch 
                    saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=self.global_epoch)

                for _ in tqdm(range(tunning_epochs), desc='Finetune Training'):

                    self._train_epoch(sess, val_finetune, batch_size, shuffle, (log_freq/10))
                    self.global_epoch += 1

                    # Save the model at each epoch 
                    saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=self.global_epoch)

        return self

    def _story_continuation(self, session, story_context, story_true_ending):
        """
        Given the story context generate an artificial ending. 

        Parameters:
        -----------
        session : tf.Session()
            A TensorFlow session 

        story_context: array-like
            Array of integers corresponding to the token IDs in the story context.  

        story_true_ending : array-like. 
            Árray of integers corresponding to the token IDs in the story ending. 

        Returns
        -------
        story_ending : array-like
            The generated story ending 
        """

        # Initial state for a single word
        states = np.zeros((2, 1, self.hidden_size))

        # Feed the story context in the RNN
        for token in story_context: 
            feed_dict = {self.one_step_word_index: [token], self.one_step_state_1: states[0], 
                         self.one_step_state_2: states[1]}

            states, _ = session.run([self.one_step_new_state, self.one_step_next_word], feed_dict=feed_dict)

        # Number of starting tokens to be used from the true ending (either 2 or 3, including <bos>)
        N = np.random.randint(low=2, high=5)

        # Initialize the generated ending 
        generated_story_ending = story_true_ending[:N].copy()

        for token in generated_story_ending:        
            feed_dict = {self.one_step_word_index: [token], self.one_step_state_1: states[0], 
                         self.one_step_state_2: states[1]}

            states, next_token = session.run([self.one_step_new_state, self.one_step_next_word], feed_dict=feed_dict)

        generated_len = len(generated_story_ending)

        while generated_len < (self.sentence_len - 1):

            next_token = next_token[0]

            # next_token = np.random.choice(self.vocab_size, p=next_token_probs)

            if next_token == self.vocab['<eos>']:
                break 

            generated_story_ending = np.append(generated_story_ending, next_token)
            generated_len += 1

            feed_dict = {self.one_step_word_index: [next_token],
                         self.one_step_state_1: states[0], self.one_step_state_2: states[1]}

            states, next_token = session.run([self.one_step_new_state, self.one_step_next_word], feed_dict=feed_dict)
            
        return decode_sentence(generated_story_ending.tolist(), self.inverse_vocab)

    def ending_generation(self, stories, true_endings):
        """
        Generate endings for the given stories

        Parameters:
        -----------
        stories : list
            Each entry in the list corresponds to an encoded story context. 

        true_endings : list
            The true (encoded) story endings. One per entry in the list.  

        Returns: 
        --------
        endings : list
            Generated endings. Each entry in the list corresponds to an ending. 
        """
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restore trained model
            saver.restore(sess,tf.train.latest_checkpoint(LOG_PATH))

            endings = []
            for story, true_ending in tqdm(zip(stories, true_endings), desc='Conditional Ending Generation'):
                ending = self._story_continuation(sess, story, true_ending)
                endings.append(ending)

        return endings
