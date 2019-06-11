# this has been inspired and based on existing tutorial on bert 
# https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy import stats

import random
import ast
import os
import sys
from os import listdir
from os.path import isfile, join

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert.run_classifier import PaddingInputExample, _truncate_seq_pair, InputFeatures


stop_words = set(stopwords.words('english'))

def create_dataset(dataset: pd.DataFrame, contains_answers=True):
    """
    Creates the dataset given a context story and two possible endings. For each ending creates a different story
    with the relative class denoting if this story is correct or not

    Parameters
    ----------
    dataset: Pandas df containing the columns InputSentence{1,2,3,4}, RandomFifthSentenceQuiz{1,2} and AnswerRightEnding

    Returns
    -------
    Pandas dataframe with the columns 'story' denoting the story context, 'ending' the candidate ending and 'class'
    signifying whether this class is correct or not
    """
    contexts = list()
    last_sentences = list()
    classes = list()
    for pos in range(len(dataset)):
        story_start = dataset.iloc[pos][['InputSentence' + str(i) for i in [1, 2, 3, 4]]].values

        contexts.append(" ".join(story_start))
        last_sentences.append(dataset.iloc[pos]['RandomFifthSentenceQuiz1'])
        contexts.append(" ".join(story_start))
        last_sentences.append(dataset.iloc[pos]['RandomFifthSentenceQuiz2'])

        if contains_answers:
            if dataset.iloc[pos]['AnswerRightEnding'] == 1:
                classes.append(0)
                classes.append(1)
            else:
                classes.append(1)
                classes.append(0)

    if contains_answers:
        return pd.DataFrame({'story': contexts, 'ending': last_sentences, 'class': classes})
    else:
        return pd.DataFrame({'story': contexts, 'ending': last_sentences})


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(flags.bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def replace_with_synonym(token, tokenizer):
    """
    Given a token, returns a random synonym of this token, given a vocabulary set (wordnet)
    """
    if token in stop_words:
        return token

    new_token = token
    synonyms = []
    for syn in wordnet.synsets(token):
        for l in syn.lemmas():
            synonyms.append(l.name())
    if len(synonyms) > 0:
        new_token = tokenizer.tokenize(random.choice(synonyms))[0]
    return new_token


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, set_synonyms=False, percentage_synonyms=0.2):
    """
    Convert a set of `InputExample`s to a list of `InputFeatures`.

    Code identical to the original model https://github.com/google-research/bert.
    Added some extra parameters to support creation of synonym sentences.
    """

    if not set_synonyms:
        percentage_synonyms = 0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, percentage_synonyms)

        features.append(feature)
    return features


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, percentage_synonyms):
    """
    Converts a single `InputExample` into a single `InputFeatures`.

    Code identical to the original model https://github.com/google-research/bert.
    Added some extra parameters to support creation of synonym sentences.
    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    # Which tokens to replace with synonyms
    set_synonyms = np.random.choice([True, False], max_seq_length,
                                    p=[percentage_synonyms, 1 - percentage_synonyms])

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    index = 1
    for token in tokens_a:
        if set_synonyms[index]:
            tokens.append(replace_with_synonym(token, tokenizer))
        else:
            tokens.append(token)
        segment_ids.append(0)
        index += 1
    tokens.append("[SEP]")
    segment_ids.append(0)
    index += 1

    if tokens_b:
        for token in tokens_b:
            if set_synonyms[index]:
                tokens.append(replace_with_synonym(token, tokenizer))
            else:
                tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def get_indices_of_last_sentence(segment_ids, input_mask, keep_story_context=False):
    """
    Based on the segment_ids return for each story in the batch the id of index of the first and last
    word corresponding to the last sentence.
    If keep_story_context is True returns as index of the first token, index 0, corresponding to the
    first word of the overall sequence.
    """
    if keep_story_context:
        # get first word of the sequence
        index_of_first_token = tf.zeros([tf.shape(segment_ids)[0]], dtype=tf.int64)
    else:
        # get the index of the first word of the last sentence
        index_of_first_token = tf.argmax(segment_ids, axis=1)

    # get the index of the last word of the last sentence
    index_of_last_token = tf.argmax((1 - input_mask) * (1 - segment_ids), axis=1) - 1

    tf_range = tf.range(tf.shape(segment_ids)[0])
    tf_range = tf.cast(tf_range, tf.int64)

    # stack with range equal to the size of the batch for easier gathering of the results
    index_of_first_token = tf.stack([tf_range, index_of_first_token], axis=1)
    index_of_last_token = tf.stack([tf_range, index_of_last_token], axis=1)

    return index_of_first_token, index_of_last_token


def dense_layer(inputs, layers, keep_prob=0.9, activation=tf.nn.relu):
    """
    Parameters
    ----------
    inputs:         tensorflow tensor of shape [BATCH_SIZE, intermediate_output_size]
    layers:         list of integers denoting the size of each layer
    keep_prob:      keep_prob parameter for tensorflow, specifying the 1 - dropout probability
    activation:     activation to use
    """

    for i, layer_size in enumerate(layers):
        dense_layer_i = tf.keras.layers.Dense(layer_size, name='DenseLayer_' + str(i), use_bias=True,
                                              activation=activation)
        inputs = dense_layer_i(inputs)
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    return inputs


def single_weight(inputs, segment_ids, weight_size):
    """
    Multiplies all relative outputs of a sequence with the same weight vector and accumulates by summing all results

    Parameters
    ----------
    inputs:         intermediate input with shape [BATCH_SIZE, SEQUENCE_LENGTH, intemrediate_output_size]
    segment_ids:    list of 0s and 1s with 1s correspodning to the position of the last sentece
    weight_size:    size of the wieght to use
    """
    assert len(
        inputs.shape) == 3, "shape for input in single_weight should be [BATCH_SIZE, SEQUENCE_LENGTH, " \
                            "intermediate_output_size] "

    # expand segmend ids to be of size [BATCH_SIZE, SEQUENCE_LENGTH, intemrediate_output_size]
    segment_ids_expanded_last_sentence = tf.tile(tf.expand_dims(segment_ids, 2), [1, 1, weight_size])
    segment_ids_expanded_last_sentence = tf.cast(segment_ids_expanded_last_sentence, tf.float32)

    # multiply with the common weight vector and only keep the outputs corresponding to the last sentence
    result_last_sentence = dense_layer(inputs, [weight_size]) * segment_ids_expanded_last_sentence

    # reduce across the SEQUENCE_LENGTH axis
    output_layer_last_sentence = tf.reduce_sum(result_last_sentence, 1)

    return output_layer_last_sentence


def bidirectional(inputs, segment_ids, input_mask, hidden_size_rnn, num_layers_fw, num_layers_bw, only_last_sentence,
                  cell_type):
    """
    Adds a bidirectional layer on top of the previous output

    Parameters
    ----------
    inputs:                 intermediate input with shape [BATCH_SIZE, SEQUENCE_LENGTH, intemrediate_output_size]
    segment_ids:            specified by bert
    input_mask:             specified by bert
    hidden_size_rnn:        hidden size for the rnn used
    num_layers_fw:          number of layers for the foward cell type chosen
    num_layers_bw:          number of layers for the backward cell type chosen
    only_last_sentence:     whether to take into accoun only the last sentence
    cell_type:              type of RNN (choose from 'lstm' and 'gru')
    """
    assert cell_type in ['gru', 'lstm'], "RNN type not supported"
    assert len(inputs.shape) == 3, "shape for input in single_weight should be [BATCH_SIZE, SEQUENCE_LENGTH, " \
                                   "intermediate_output_size] "

    hidden_size = inputs.shape[-1].value
    
    if only_last_sentence:
        # keep only the outputs that correspond to the last sentence
        inputs = inputs * tf.tile(
            tf.expand_dims(tf.cast(segment_ids, tf.float32), 2), [1, 1, hidden_size])

    if cell_type == 'gru':
        cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=hidden_size_rnn) for _ in range(num_layers_fw)]
        cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=hidden_size_rnn) for _ in range(num_layers_bw)]
    elif cell_type == 'lstm':
        cells_fw = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size_rnn) for _ in range(num_layers_fw)]
        cells_bw = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size_rnn) for _ in range(num_layers_bw)]
    else:
        print('RNN cell type', cell_type, 'not supported')
        sys.exit(1)

    # we stack the cells together and create one big RNN cell
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)

    # make input to the required form for the rnn
    inputs = tf.transpose(inputs, [1, 0, 2])
    inputs = tf.unstack(inputs, num=flags.max_seq_length)

    with tf.variable_scope("last_sentence"):
        outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(cell_fw,
                                                                                   cell_bw,
                                                                                   inputs,
                                                                                   dtype=tf.float32)

    # stack outpts and transpose them back into shape [BATCH_SIZE, SEQUENCE_LENGTH, hidden_size * 2]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    if only_last_sentence:
        # get the outputs that correspond to the first and last words of the last sentence only
        """ 
        Although we have already eliminated the rest of the words in a previous step, the relative 
        large size of the word sequence would mean that the information will easily get lost. 
        At this step, we take the output of the RNN at the steps where the most information is present.
        """
        index_of_first_token, index_of_last_token = get_indices_of_last_sentence(segment_ids, input_mask,
                                                                                 keep_story_context=False)
    else:
        # in case the whole sentence is taken into account, outputs at positions first and last word of the sequence
        index_of_first_token, index_of_last_token = get_indices_of_last_sentence(segment_ids, input_mask,
                                                                                 keep_story_context=True)

    first_token_output = tf.gather_nd(outputs, index_of_first_token)
    last_token_output = tf.gather_nd(outputs, index_of_last_token)

    return tf.concat([first_token_output, last_token_output], 1)


def conv(inputs, kernel_sizes, filters, pool_sizes, max_seq_length, hidden_size):
    """
    Adds a conv2d layer on top of the previous output

    Parameters
    ----------
    inputs:                 intermediate input with shape [BATCH_SIZE, SEQUENCE_LENGTH, intermediate_output_size]
    kernel_sizes:           list of list of integer, every list contains two integers, the kernel size of each layer
    filters:                number of filters for each convolution layer
    pool_sizes:             pool size for every layer
    max_seq_length:         max length of a sequence
    hidden_size:            original hidden size
    """
    assert len(inputs.shape) == 3, "shape for input in single_weight should be [BATCH_SIZE, SEQUENCE_LENGTH, " \
                                   "intemrediate_output_size] "
    assert len(kernel_sizes) == len(filters) and len(filters) == len(
        pool_sizes), "incompatible number of layers for kernel_sizes, filters and pool_sizes"

    # expand dims, required by tf.layers.conv2d
    inputs = tf.expand_dims(inputs, 3)

    # explicitly define the shape of the tensor
    inputs = tf.reshape(inputs, [-1, max_seq_length, hidden_size, 1])

    for i in range(len(filters)):
        layer_filter = filters[i]
        layer_kernel = kernel_sizes[i]
        layer_pool = pool_sizes[i]

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=layer_filter,
            kernel_size=[layer_kernel[0], layer_kernel[1]],
            padding="same",
            activation=tf.nn.relu)

        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[layer_pool, layer_pool], strides=layer_pool)

    # transform to a tensor of shape [BATCH_SIZE, intermediate_hidden_size]
    return tf.reshape(inputs, [-1, inputs.get_shape()[1] * inputs.get_shape()[2] * inputs.get_shape()[3]])


def highway_network(inputs, num_highway_layers):
    """
    Hghway network as specified from https://arxiv.org/abs/1505.00387
    """

    hidden_size = inputs.shape[-1].value

    for i in range(num_highway_layers):
        # first layer
        x_gate = dense_layer(inputs, [hidden_size], activation=tf.nn.sigmoid)
        x_project = dense_layer(inputs, [hidden_size], activation=tf.nn.relu)

        inputs = x_gate * x_project + (1 - x_gate)

    return inputs


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
        flags.bert_model_hub,
        trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    output_layer = bert_outputs["pooled_output"]
    inputs = bert_outputs['sequence_output']

    # network architecture: SingleWeight-3072:
    network_architecture = flags.network

    if network_architecture == "None":
        final_output = output_layer
    else:
        for network_layer in network_architecture.split(':'):
            try:
                network_layer_parameters = network_layer.split('-')
                network_type = network_layer_parameters[0]

                if network_type == 'singleweight':
                    # network_type='singleweight'-weight_size
                    weight_size = int(network_layer_parameters[1])

                    if flags.verbose:
                        print('Creating layer singleweight:', weight_size)

                    inputs = single_weight(inputs, segment_ids, weight_size)

                elif network_type == 'bidirectional':
                    # network_type='bidirectional'-hidden_size-num_layers_fw-num_layers_bw-{True|False}-{lstm|gru}
                    hidden_size_rnn = int(network_layer_parameters[1])
                    num_layers_fw = int(network_layer_parameters[2])
                    num_layers_bw = int(network_layer_parameters[3])
                    only_last_sentence = bool(network_layer_parameters[4])
                    cell_type = network_layer_parameters[5]

                    if flags.verbose:
                        print('Creating layer bidirectional:', hidden_size_rnn, num_layers_fw, num_layers_bw,
                              only_last_sentence, cell_type)

                    inputs = bidirectional(inputs, segment_ids, input_mask, hidden_size_rnn, num_layers_fw,
                                           num_layers_bw, only_last_sentence, cell_type)

                elif network_type == 'conv':
                    # network_type='conv'-[kernel_sizes]-[filters]-[pool_sizes]

                    """
                    At this point we should mention that using a concvolution layer of top of the transformer outputs
                    can be very unstable. We got our best results using two layers of kelrnel sizes [5,5] and [5,5] 
                    followed by pooling layers of size [2, 2] and stride 2. 
                    Unstable results are mainly the small training set used in this case. 
                    To alleviate we train we a smaller learning rate (a value of 5e-6 or smaller is recommended) and for
                    a slightly larger number of epochs (5 to 6).
                    Finally we should mention that although some of the models in the ensemble may encounter problems in
                    training and output worse than average or even random results (depending on the architecture this 
                    can be avoided), the overall final predictions are not significantly influenced 
                    """
                    kernel_sizes = ast.literal_eval(network_layer_parameters[1])
                    filters = ast.literal_eval(network_layer_parameters[2])
                    pool_sizes = ast.literal_eval(network_layer_parameters[3])

                    if flags.verbose:
                        print('Creating layer conv:', kernel_sizes, filters, pool_sizes)

                    inputs = conv(inputs, kernel_sizes, filters, pool_sizes,
                                  flags.max_seq_length, inputs.shape[-1].value)

                elif network_type == 'highway':
                    # network_type='highway'-[num_layers]
                    num_layers = int(network_layer_parameters[1])

                    if flags.verbose:
                        print('Creating layer highway:', num_layers)

                    highway_network(inputs, num_layers)
                elif network_type == 'dense':
                    # network_type='dense'-layers-keep_prob

                    layers = ast.literal_eval(network_layer_parameters[1])
                    keep_prob = float(network_layer_parameters[2])

                    if flags.verbose:
                        print('Creating layer dense:', layers, keep_prob)

                    inputs = dense_layer(inputs, layers, keep_prob=keep_prob)
                else:
                    print('Network type', network_type, 'not supported')
                    sys.exit(1)

            except IndexError:
                print('Invalid network architecture for network', network_layer)
                sys.exit(1)
            except SyntaxError:
                print('Error while evaluating list from string', network_layer)
                sys.exit(1)

        final_output = inputs

    # project to the number of available classes
    final_output = tf.nn.dropout(final_output, keep_prob=0.9)
    logits = tf.layers.dense(final_output, num_labels, use_bias=True)

    with tf.variable_scope("loss"):
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return predicted_labels, log_probs

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, predicted_labels, log_probs


def get_final_predictions(in_contexts, in_last_sentences, tokenizer, estimator: tf.estimator.Estimator, label_list):
    """
    Return the log probabilities based on the story context and the endings proposed

    Parameters
    ----------
    in_contexts:            str of the story context
    in_last_sentences:      proposed last sentence
    tokenizer:              bert tokenizer
    estimator:              tf.estimator
    label_list:             possible values
    """
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=y, label=0) for x, y in
                      zip(in_contexts, in_last_sentences)]  # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, flags.max_seq_length,
                                                                 tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=flags.max_seq_length,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    predictions = [prediction['probabilities'] for prediction in predictions]

    return predictions


# noinspection PyTypeChecker
def combine_predictions(predictions):
    """
    Return the most probable ending

    Parameters
    ----------
    predictions:    Array of size [2*num_predictions, 2]. Rows i, i + 1 for i even, correspond to the probabilities
                    that the correct ending is ending 1 for the row i and ending 2 for the row i + 1
    """
    my_predictions = []

    i = 0
    while i < len(predictions):

        # take probabilities from log predictions
        p_first = np.exp(predictions[i])
        p_second = np.exp(predictions[i + 1])

        p1 = p_first[0] + p_second[1]
        p2 = p_first[1] + p_second[0]

        if p1 > p2:
            my_predictions.append(1)
        else:
            my_predictions.append(2)
        i += 2

    return np.array(my_predictions)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    # noinspection PyUnusedLocal
    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


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

    data_val = pd.read_csv(os.path.join(flags.data_dir, flags.val_file_name), header='infer')
    data_test = pd.read_csv(os.path.join(flags.data_dir, flags.test_file_name), header='infer')
    data_test_eth = pd.read_csv(os.path.join(flags.data_dir, flags.provided_test_file_name), header='infer')

    # download required nltk packages
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # create datasets based on the stories provided
    train = shuffle(create_dataset(data_val))
    test = create_dataset(data_test)
    test_eth = create_dataset(data_test_eth, contains_answers=False)

    print('Size of the training set:\t', len(train))
    print('Size of the test set:\t\t', len(test))
    print('Size of the test set eth:\t', len(test_eth))

    # transforms the dataset into a form that bert can understand
    CONTEXT_COLUMN = 'story'
    ENDING_COLUMN = 'ending'
    LABEL_COLUMN = 'class'

    # label_list is 0 for a true story and 1 for a false story
    label_list = [0, 1]

    train_InputExamples = pd.concat([train] * flags.num_epochs).apply(
        lambda x:bert.run_classifier.InputExample(guid=None,
                                                  text_a=x[CONTEXT_COLUMN],
                                                  text_b=x[ENDING_COLUMN],
                                                  label=x[LABEL_COLUMN]),
        axis=1)

  
    # get tokenizer from bert
    tokenizer = create_tokenizer_from_hub_module()

    # Convert our train and test features to InputFeatures that BERT understands.
    # replace in each tarining sample a number of words
    train_features = convert_examples_to_features(train_InputExamples, label_list, flags.max_seq_length, tokenizer,
                                                  set_synonyms=True, percentage_synonyms=flags.percentage_synonyms)

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / flags.batch_size)
    num_warmup_steps = int(num_train_steps * flags.warmup_proportion)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=flags.output_dir,
        save_summary_steps=flags.save_summary_steps,
        save_checkpoints_steps=flags.save_checkpoints_steps)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=flags.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": flags.batch_size})

    # ==============================================================================================================
    # start training

    # First remove previous records of predictions
    os.system('rm -rf ' + flags.save_results_dir + ' || true')
    os.system('mkdir ' + flags.save_results_dir)

    true_labels_test = test['class'].values[::2] + 1

    for i in range(flags.num_estimators):
        os.system('rm -rf ' + flags.output_dir + ' || true')

        train_features = shuffle(train_features)

        # Create an input function for training
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=flags.max_seq_length,
            is_training=True,
            drop_remainder=False)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        predictions = get_final_predictions(test['story'].values, test['ending'].values,
                                            tokenizer, estimator, label_list)

        test_score = accuracy_score(true_labels_test, combine_predictions(predictions))

        # save results on disk to combine results at a next step
        # noinspection PyTypeChecker
        np.savetxt(
            os.path.join("./" + flags.save_results_dir,
                         "predictions_original_test_" + str(test_score) + '_classifier_' + str(i) + '.csv'),
            predictions, delimiter=",")

        predictions_eth = get_final_predictions(test_eth['story'].values, test_eth['ending'].values,
                                                tokenizer, estimator, label_list)

        np.savetxt(
            os.path.join("./" + flags.save_results_dir,
                         "predictions_test_eth_classifier_" + str(i) + '.csv'),
            predictions_eth, delimiter=",")

    # ==============================================================================================================
    # finally combine results for the ensemble score

    # Combine all previous acquired results for an ensemble
    files = [f for f in listdir(flags.save_results_dir) if isfile(join(flags.save_results_dir, f))]

    # take number of classifiers from the names of the file
    # noinspection PyTypeChecker
    classifiers = [int(file.split("_")[5].split(".")[0]) for file in files if 'original' in file]
    num_classifiers = np.max(classifiers)

    predictions_test = list()
    predictions_eth_test_set = list()

    for i in range(num_classifiers + 1):
        roc_test_file = [x for x in files if 'classifier_' + str(i) in x and 'original' in x][0]
        eth_test_file = [x for x in files if 'classifier_' + str(i) in x and 'eth' in x][0]

        accuracy = float(roc_test_file.split('_')[3])
        
        predictions_file_test = np.genfromtxt(os.path.join(flags.save_results_dir, roc_test_file), delimiter=',')
        predictions_test.append(predictions_file_test)

        predictions_file_eth_test_set = np.genfromtxt(os.path.join(flags.save_results_dir, eth_test_file), delimiter=',')
        predictions_eth_test_set.append(predictions_file_eth_test_set)

        print(f'For classifier {i:2d} roc test accuracy {accuracy:.6f}')

    def print_ensemble_predictions(predictions, true_labels=None, original_test_set=True):
        assert (original_test_set==False or true_labels is not None), "If predictions on roc stories specify labels."

        preds_mode = [combine_predictions(p) for p in predictions]
        preds_mode = np.array(preds_mode)
        preds_mode = stats.mode(preds_mode)[0][0]

        if original_test_set:
            print('ensemble accuracy by taking the mode of each prediction')
            print(accuracy_score(true_labels, preds_mode))
        else:
            np.savetxt(os.path.join(flags.save_results_dir, "predictions_test_eth_ensemble_mode.csv"),
                            preds_mode, delimiter=",")

        preds_prob = np.mean(predictions, axis=0)
        preds_prob = combine_predictions(preds_prob)

        if original_test_set:
            print('ensemble accuracy by adding the prediction probabilities')
            print(accuracy_score(true_labels, preds_prob))
        else:
            np.savetxt(os.path.join(flags.save_results_dir, "predictions_test_eth_ensemble_probabilities.csv"),
                            preds_mode, delimiter=",")

    print_ensemble_predictions(predictions_test, true_labels=true_labels_test)
    print_ensemble_predictions(predictions_eth_test_set, original_test_set=False)


if __name__ == '__main__':

    # delete flags namespace to avoid flag duplicate errors, as it is also used in bert tensorflow_hub
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.flags.FLAGS)

    tf.app.flags.DEFINE_string("data_dir", "/cluster/home/sanagnos/NLU/project2/data/",
                               "where the training data is stored")
    tf.app.flags.DEFINE_string("val_file_name", "cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
                               "name of the validation file")
    tf.app.flags.DEFINE_string("test_file_name", "test_for_report-stories_labels.csv",
                               "name of the original test cloze file")
    tf.app.flags.DEFINE_string("provided_test_file_name", "test-stories.csv",
                               "name of the provided test file")

    tf.app.flags.DEFINE_string("output_dir", "./output_dir", "Where the output data will be stored")
    tf.app.flags.DEFINE_string("tfhub_cache_dir", "./tfhub_cache_dir", "Cached directory for tf hub")
    tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs to run for")
    tf.app.flags.DEFINE_string("bert_model_hub", "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1",
                               "BERT model to choose")
    tf.app.flags.DEFINE_integer("batch_size", 16, "batch size")
    tf.app.flags.DEFINE_float("learning_rate", 2e-5, "learning rate")
    tf.app.flags.DEFINE_float("percentage_synonyms", 0.2,
                              "percentage of words to replace with synonyms in each story")

    # Warmup is a period of time where the learning rate
    # is small and gradually increases (usually helps training))
    tf.app.flags.DEFINE_float("warmup_proportion", 0.1, "warmup proportion")

    tf.app.flags.DEFINE_integer("save_checkpoints_steps", 5000, "save model every these many steps")
    tf.app.flags.DEFINE_integer("save_summary_steps", 50, "save summary every these many steps")
    tf.app.flags.DEFINE_string("save_results_dir", "./results_predictions", "directory to store intermediate results")

    tf.app.flags.DEFINE_integer("num_estimators", 15, "number of estimators to use for the ensemble")
    tf.app.flags.DEFINE_integer("max_seq_length", 96, "Maximum length of a sequence of words in a story.")
    tf.app.flags.DEFINE_boolean("verbose", False, "vebose")
    tf.app.flags.DEFINE_string('f', '', 'kernel')  # Dummy entry because colab is weird.

    tf.app.flags.DEFINE_string('network',
                               "bidirectional-768-1-1-True-lstm:dense-[512]-0.9",
                               "network architecture: passed-form: network_type:network_type:...:network_type \n \
                                where each network type has the form network_name-parameter_1-parameter_2-...-parameter_N \n \
                                Supported options are: \n \
                                singleweight-weight_size \n \
                                bidirectional-hidden_size-num_layers_fw-num_layers_bw-{only_last_sentence:True|False}-{lstm|gru} \n \
                                conv-[kernel_sizes]-[filters]-[pool_sizes] \n \
                                'highway'-[num_layers] \n \
                                'dense'-layers-keep_prob (default bidirectional-768-1-1-True-lstm:dense-[512]-0.9)")

    flags = tf.app.flags.FLAGS

    tf.app.run()
