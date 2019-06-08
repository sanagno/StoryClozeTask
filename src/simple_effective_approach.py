# implementation of the models specified in A Simple and Effective Approach to the Story Cloze Test
# https://www.aclweb.org/anthology/N18-2015

import pandas as pd
import os
from model import NLUModel
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import random
import math
import shutil

ROC_TRAIN_SET = 'train_stories.csv'
ROC_VAL_SET = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
ROC_TEST_SET = 'test_for_report-stories-labels.csv'

TRAIN_SKIP_THOUGHTS_EMBEDDINGS = '/cluster/project/infk/courses/machine_perception_19/' \
                                 'Sasglentamekaiedo/skip-thoughts-embbedings.npy'
VAL_SKIP_THOUGHTS_EMBEDDINGS = '/cluster/project/infk/courses/machine_perception_19/' \
                               'Sasglentamekaiedo/skip-thoughts-embbedings_validation.npy'
TEST_SKIP_THOUGHTS_EMBEDDINGS = '/cluster/project/infk/courses/machine_perception_19/' \
                                'Sasglentamekaiedo/skip-thoughts-embbedings_test.npy'


def create_dataset_from_embeddings(embeddings, df):
    """
    For each story and possible endings creates two separate stories with the two endings, one correct and one wrong.

    Parameters
    ----------
    embeddings:     For each story a array [6, skip_thought_embedding_size]
    df:             Pandas df containing the stories in order to figure out which of the possible endings in the correct
    """
    v_embeddings = list()
    v_classes = list()
    correct_answers = df['AnswerRightEnding'].values

    for i, story_embedding in enumerate(embeddings):
        v_embeddings.append(np.append(story_embedding[:4], [story_embedding[4]], axis=0))
        v_embeddings.append(np.append(story_embedding[:4], [story_embedding[5]], axis=0))

        if correct_answers[i] == 1:
            v_classes.append(0)
            v_classes.append(1)
        else:
            v_classes.append(1)
            v_classes.append(0)

    return np.array(v_embeddings), np.array(v_classes)


def get_final_predictions(probabilities):
    """
    From the probabilities of the endings, choose as the correct ending the one that has the higher probability
    """
    my_predictions = []

    probabilities_exp = np.exp(probabilities)

    i = 0
    while i < len(probabilities):
        p_first = probabilities_exp[i]
        p_second = probabilities_exp[i + 1]

        p1 = p_first[0] + p_second[1]
        p2 = p_first[1] + p_second[0]

        if p1 > p2:
            my_predictions.append(0)
        else:
            my_predictions.append(1)
        i += 2

    return np.array(my_predictions)


class Encoder(tf.keras.Model):
    # Creates an encoder based on a GRU cell
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    # Attention on top of an encoder
    # taken from https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units, name='Dense_1')
        self.W2 = tf.keras.layers.Dense(units, name='Dense_2')
        self.V = tf.keras.layers.Dense(1, name='Dense_3')

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class DenseLayerWithSoftmax(tf.keras.Model):
    # creates a dense layer with the layers specified, followed by a softmax layer at the end for the two classes
    def __init__(self, layers, num_classes, dropout_keep_proba=0.9, activation=tf.nn.relu):
        super(DenseLayerWithSoftmax, self).__init__()

        self.dense_layers = []
        self.dropout_keep_proba = dropout_keep_proba
        self.num_classes = num_classes

        for i, layer_size in enumerate(layers):
            self.dense_layers.append(
                tf.keras.layers.Dense(layer_size, name='DenseLayer_' + str(i), use_bias=True, activation=tf.nn.relu))

        self.final_layer = tf.keras.layers.Dense(self.num_classes, name='DenseLayer_final', use_bias=True)

    def call(self, logits, input_labels):
        for layer in self.dense_layers:
            logits = layer(logits)
            logits = tf.nn.dropout(logits, keep_prob=self.dropout_keep_proba)

        logits = self.final_layer(logits)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(input_labels, depth=self.num_classes, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

        per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return predicted_labels, loss, log_probs


class FCSkip(tf.keras.Model):
    def __init__(self, units, num_classes=2, fc_layers=[], dropout_keep_prob=0.9, activation=tf.nn.relu):
        super(FCSkip, self).__init__()

        self.units = units

        self.encoder = Encoder(self.units)
        self.feed_forward = DenseLayerWithSoftmax(fc_layers, num_classes,
                                                  dropout_keep_proba=dropout_keep_prob, activation=activation)

    def call(self, input_embeddings, input_labels):
        # sample input
        sample_hidden = self.encoder.initialize_hidden_state(tf.shape(input_embeddings)[0])
        sample_output, sample_hidden = self.encoder(input_embeddings[:, :4, :], sample_hidden)

        # concatenated_input = tf.concat([sample_output[:, -1, :], input_embeddings[:, 4, :]], axis=1)
        concatenated_input = sample_output[:, -1, :] + input_embeddings[:, 4, :]

        return self.feed_forward(concatenated_input, input_labels)


class LSSkip(tf.keras.Model):
    def __init__(self, units, num_classes=2, fc_layers=[], dropout_keep_prob=0.9, activation=tf.nn.relu):
        super(LSSkip, self).__init__()

        self.units = units

        self.feed_forward = DenseLayerWithSoftmax(fc_layers, num_classes,
                                                  dropout_keep_proba=dropout_keep_prob, activation=activation)

    def call(self, input_embeddings, input_labels):

        # concatenated_input = tf.concat([input_embeddings[:, 3, :], input_embeddings[:, 4, :]], axis=1)
        concatenated_input = input_embeddings[:, 3, :] + input_embeddings[:, 4, :]

        return self.feed_forward(concatenated_input, input_labels)


# noinspection PyPep8Naming
class SimpleAndEffectiveApproach(NLUModel):
    # Parameters in functions fix and predict are unused. This is to avoid calculating Skip thought embeddings every
    # times, but use already calculated embeddings.

    train_embeddings = None
    train_classes = None
    test_embeddings = None
    test_classes = None
    final_test_embeddings = None
    final_test_classes = None
    batch_size = None

    def __init__(self, units, fc_layers=None, num_classes=2, train_on_validation=False, mode='FC-skip', verbose=False,
                 negative_sampling=3, learning_rate=1e-3):
        assert mode in ['FC-skip', 'LS-skip'], "mode specified not supported"

        super(SimpleAndEffectiveApproach, self).__init__('SimpleAndEffectiveApproach')
        self.train_on_validation = train_on_validation
        self.verbose = verbose
        self.negative_sampling = negative_sampling
        self.mode = mode
        self.fc_layers = fc_layers
        self.units = units
        self.learning_rate = learning_rate

    def _create_graph(self):
        """
        Creates the graph as specified by the parameters in the init function
        """
        train_x, train_y, test_x, test_y, final_test_x, final_test_y = self._prepare_embeddings()

        if self.mode == 'FC-skip':
            if self.fc_layers is None:
                print('Using default values for feed forward network [256, 64]')
                self.fc_layers = [256, 64]
            fc_skip = FCSkip(self.units, num_classes=2, fc_layers=self.fc_layers,
                             dropout_keep_prob=0.9, activation=tf.nn.relu)

            self.predicted_labels_train, self.loss_train, self.log_probs_train = fc_skip(train_x, train_y)
            self.predicted_labels_test, self.loss_test, self.log_probs_test = fc_skip(test_x, test_y)
            self.predicted_labels_final_test, self.loss_final_test, self.log_probs_final_test = fc_skip(final_test_x,
                                                                                                        final_test_y)

        elif self.mode == 'LS-skip':
            if self.fc_layers is None:
                print('Using default values for feed forward network [2400, 1200, 600]')
                self.fc_layers = [2400, 1200, 600]

            ls_skip = LSSkip(self.units, num_classes=2, fc_layers=self.fc_layers, dropout_keep_prob=0.9,
                             activation=tf.nn.relu)

            self.predicted_labels_train, self.loss_train, self.log_probs_train = ls_skip(train_x, train_y)
            self.predicted_labels_test, self.loss_test, self.log_probs_test = ls_skip(test_x, test_y)
            self.predicted_labels_final_test, self.loss_final_test, self.log_probs_final_test = ls_skip(final_test_x,
                                                                                                        final_test_y)

    def _prepare_embeddings(self):
        if not self.train_on_validation:
            data_train = pd.read_csv(os.path.join(flags.data_dir, ROC_TRAIN_SET), header='infer')
            # has a shape (88161, 5, 4800)
            train_skip_thought = np.load(TRAIN_SKIP_THOUGHTS_EMBEDDINGS)

        data_val = pd.read_csv(os.path.join(flags.data_dir, ROC_VAL_SET), header='infer')
        # has a shape (1871, 6, 4800)
        validation_skip_thought = np.load(VAL_SKIP_THOUGHTS_EMBEDDINGS)

        data_test = pd.read_csv(os.path.join(flags.data_dir, ROC_TEST_SET), header='infer')
        # has a shape (1871, 6, 4800)
        test_skip_thought = np.load(TEST_SKIP_THOUGHTS_EMBEDDINGS)

        # create set for validation dataset
        val_embeddings, val_classes = create_dataset_from_embeddings(validation_skip_thought, data_val)
        self.final_test_embeddings, self.final_test_classes = create_dataset_from_embeddings(test_skip_thought,
                                                                                             data_test)

        if self.train_on_validation:
            self.train_embeddings, self.train_classes = val_embeddings, val_classes

            self.test_embeddings, self.test_classes = self.final_test_embeddings, self.final_test_classes
        else:
            # noinspection PyUnboundLocalVariable
            self.train_embeddings, self.train_classes = train_skip_thought, np.zeros(len(train_skip_thought))

            self.test_embeddings, self.test_classes = val_embeddings, val_classes

        self.num_samples_training = len(self.train_embeddings)
        self.num_samples_test = len(self.test_embeddings)
        self.num_samples_final_test = len(self.final_test_embeddings)

        if self.train_on_validation:
            if self.verbose:
                print('Loading without negative sampling.')

            train_x, train_y = self.create_dataset(0, self.batch_size, 0)
        else:
            if self.verbose:
                print('Loading with negative sampling on the original training set.')

            train_x, train_y = self.create_dataset(0, self.batch_size, self.negative_sampling)

        test_x, test_y = self.create_dataset(1, self.batch_size, 0)

        final_test_x, final_test_y = self.create_dataset(2, self.batch_size, 0)

        return train_x, train_y, test_x, test_y, final_test_x, final_test_y

    @staticmethod
    def sample_negatives(embeddings, embeddings_batch, classes_batch, negative_sampling):
        new_classes = []
        new_embeddings_batch = []
        for i, embedding in enumerate(embeddings_batch):

            new_embeddings_batch.append(embedding)
            new_classes.append(classes_batch[i])

            for _ in range(negative_sampling):
                new_embeddings_batch.append(
                    np.concatenate((embedding[:4], [random.choice(embeddings[:, 4, :])]), axis=0))
                # negative class always
                new_classes.append(1)
        return np.array(new_embeddings_batch, dtype=np.float32), np.array(new_classes, dtype=np.int32)

    def generator(self, mode, batch_size=64, negative_sampling=0):
        """
        negative_sampling: For each positive sample these many negatives
        We use this generator to obtain all the negative samples
        Also avoids storing all tf.Dataset in the tf.Graph

        Parameters
        ----------
        mode:               0: use training embeddings, 1: use test embeddings, 2: use final test embeddings
        negative_sampling:  how many times sample each sentence
        """
        if mode == 0:
            embeddings = self.train_embeddings
            classes = self.train_classes
        elif mode == 1:
            embeddings = self.test_embeddings
            classes = self.test_classes
        else:
            embeddings = self.final_test_embeddings
            classes = self.final_test_classes

        if negative_sampling > 0:
            batch_size /= (negative_sampling + 1)
            if batch_size != int(batch_size):
                raise Exception('Batch size should be an integer. Please change negative sampling rate')

            batch_size = int(batch_size)

        # repeat
        while True:
            if mode == 0:
                embeddings, classes = shuffle(embeddings, classes)

            length = len(embeddings)
            for ndx in range(0, length, batch_size):
                embeddings_batch = embeddings[ndx: min(ndx + batch_size, length)]
                classes_batch = classes[ndx: min(ndx + batch_size, length)]

                if negative_sampling <= 0:
                    yield embeddings_batch, classes_batch
                else:
                    yield self.sample_negatives(embeddings, embeddings_batch, classes_batch, negative_sampling)

    def create_dataset(self, mode, batch_size, negative_sampling):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32),
                                                 output_shapes=(
                                                     tf.TensorShape([None, 5, 4800]), tf.TensorShape([None])),
                                                 args=([mode, batch_size, negative_sampling]))

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def _predict_wth_session(self, sess, final_test=False):
        predictions = []

        if final_test:
            num_samples = self.num_samples_final_test
            log_probs = self.log_probs_final_test
        else:
            num_samples = self.num_samples_test
            log_probs = self.log_probs_test

        for _ in range(math.ceil(num_samples / self.batch_size)):
            predictions_batch = sess.run(log_probs)

            predictions.append(predictions_batch)

        return np.concatenate(predictions, axis=0).reshape(-1, 2)

    def predict(self, X):
        """
        Loads last model and predicts on the final test set
        X is unused
        """
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(flags.log_path))

            return self._predict_wth_session(sess, final_test=True)

    def fit(self, X, y, epochs=10, batch_size=64):
        """
        X, y are unused in this case
        """

        self.batch_size = batch_size
        self._create_graph()

        update_learning_rate = 20  # number of times to update the learning rate, empirical
        if self.train_on_validation:
            update_lr_every = int((math.ceil(self.num_samples_training / batch_size) * epochs) / update_learning_rate)
        else:
            update_lr_every = int(
                (math.ceil(self.num_samples_training / (
                            batch_size / (self.negative_sampling + 1))) * epochs) / update_learning_rate)

        global_step = tf.Variable(0, trainable=False, name="global_step")
        if self.verbose:
            print('Updating the learning rate every:', update_lr_every, 'steps.')

        # learning rate 1e-3 for most models
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,  # Base learning rate.
            global_step,  # Current index into the dataset.
            update_lr_every,  # Decay step.
            0.96,  # Decay rate.
            staircase=True)

        # exponential decay on the learning rate and clipping of gradients usually helps
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_train))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # define model saver
            saver = tf.train.Saver(tf.global_variables())

            if self.train_on_validation:
                number_of_steps = math.ceil(self.num_samples_training / batch_size)
            else:
                number_of_steps = math.ceil(self.num_samples_training / (batch_size / (self.negative_sampling + 1)))

            # used for displaying the progress of tarining at the end of each epoch
            last_epoch = 0

            for i in range(number_of_steps * epochs):
                epoch_cur = i // number_of_steps

                sess.run([train_op, global_step])

                if epoch_cur > last_epoch and self.verbose:
                    last_epoch = epoch_cur

                    log_probs = self._predict_wth_session(sess)
                    score = accuracy_score(self.test_classes[::2], get_final_predictions(log_probs))
                    print('At epoch %3d score on validation set %.4f' % (last_epoch, score))

            saver.save(sess, os.path.join(flags.log_path, "model"),
                       global_step=int(epochs * self.num_samples_training / batch_size))


def main(argv):
    print("\nCommand-line Arguments:")
    for key in flags.flag_values_dict():
        if key == 'f':
            continue
        print("{:<22}: {}".format(key.upper(), flags[key].value))
    print(" ")

    # suppress some deprecation warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    device_name = tf.test.gpu_device_name()

    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')

    ###

    if os.path.exists(flags.log_path) and os.path.isdir(flags.log_path):
        shutil.rmtree(flags.log_path)
    os.makedirs(flags.log_path, exist_ok=True)

    if flags.train_on_validation:
        model = SimpleAndEffectiveApproach(flags.units, train_on_validation=True,
                                           mode=flags.mode, verbose=flags.verbose, learning_rate=flags.learning_rate)
    else:
        model = SimpleAndEffectiveApproach(flags.units, train_on_validation=False,
                                           mode=flags.mode, verbose=flags.verbose, learning_rate=flags.learning_rate)

    model.fit(None, None, batch_size=flags.batch_size, epochs=flags.num_epochs)

    log_predictions = model.predict(None)

    score = accuracy_score(model.final_test_classes[::2], get_final_predictions(log_predictions))
    print('Final score on test set based on last epoch model %.5f' % score)


if __name__ == '__main__':

    tf.app.flags.DEFINE_string("log_path", "../log_path", "Path to logging directory")
    tf.app.flags.DEFINE_string("data_dir", '../data/', "Where the training data is stored")
    tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs to run for")
    tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    tf.app.flags.DEFINE_boolean("verbose", True, "verbose")

    tf.app.flags.DEFINE_integer("units", 4800, "Unis in rnn cell. Should be equal to skip thoughts embedding size")
    tf.app.flags.DEFINE_integer("train_on_validation", 0, "Train on the validation set or not")
    tf.app.flags.DEFINE_string("mode", 'FC-skip', "Training mode ('FC-skip', 'LS-skip')")

    tf.app.flags.DEFINE_string('f', '', 'kernel')  # Dummy entry because colab is weird.
    flags = tf.app.flags.FLAGS

    tf.app.run()

