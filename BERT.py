
# coding: utf-8

# ## Import dataset

# In[1]:


import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress some deprecation warnings

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')


# In[3]:


import pandas as pd
import os

data_directory = './data'

# data_train = pd.read_csv(os.path.join(data_directory, 'train_stories.csv'), header='infer')

data_val = pd.read_csv(os.path.join(data_directory, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'), header='infer')
data_test = pd.read_csv(os.path.join(data_directory, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'), header='infer')


# In[4]:


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import numpy as np


# In[5]:


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization



OUTPUT_DIR = 'output_dir'


import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# In[8]:

def create_dataset(data_val):
    contexts = list()
    last_sentences = list()
    classes = list()
    for pos in range(len(data_val)):
        story_start = data_val.iloc[pos][['InputSentence' + str(i) for i in [1, 2, 3, 4]]].values
        
        contexts.append(" ".join(story_start))
        last_sentences.append(data_val.iloc[pos]['RandomFifthSentenceQuiz1'])
        contexts.append(" ".join(story_start))
        last_sentences.append(data_val.iloc[pos]['RandomFifthSentenceQuiz2'])
        
        if data_val.iloc[pos]['AnswerRightEnding'] == 1:
            classes.append(0)
            classes.append(1)
        else:
            classes.append(1)
            classes.append(0)
            
    return pd.DataFrame({'story': contexts, 'ending': last_sentences, 'class': classes})

val_pd = create_dataset(data_val)
test_pd = create_dataset(data_test)


# In[9]:

from sklearn.utils import shuffle

train = shuffle(val_pd)
train_unshuffled = val_pd
test = test_pd

print(len(train))
print(len(test))

# In[10]:

CONTEXT_COLUMN = 'story'
ENDING_COLUMN = 'ending'
LABEL_COLUMN = 'class'

# label_list is 0 for a true story and 1 for a false story
label_list = [0, 1]

REPLICATION_FACTOR = 3

train_InputExamples = pd.concat([train]*REPLICATION_FACTOR).apply(lambda x: 
                                                                  bert.run_classifier.InputExample(guid=None,
                                                                  text_a = x[CONTEXT_COLUMN], 
                                                                  text_b = x[ENDING_COLUMN], 
                                                                  label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[CONTEXT_COLUMN], 
                                                                   text_b = x[ENDING_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

# In[11]:


# This is a path to an uncased (all lowercase) version of BERT
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

from bert.run_classifier import PaddingInputExample, _truncate_seq_pair, InputFeatures
import nltk
from nltk.corpus import wordnet
import random

def replace_with_synonym(token, tokenizer):
  new_token = token
  synonyms = []
  for syn in wordnet.synsets(token):
    for l in syn.lemmas():
      synonyms.append(l.name())
  if len(synonyms) > 0:
    new_token = tokenizer.tokenize(random.choice(synonyms))[0]
#     print(token, new_token)
  return new_token


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, set_synonyms=False, percentage_synonyms=0.2):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""
  
  if set_synonyms == False:
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
  """Converts a single `InputExample` into a single `InputFeatures`."""

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


# In[12]:

# We'll set sequences to be at most this tokens long.
MAX_SEQ_LENGTH = 96
# Convert our train and test features to InputFeatures that BERT understands.
train_features = convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer,
                                              set_synonyms=True, percentage_synonyms=0.2)

test_features = convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# In[13]:


tf.reset_default_graph()

class DenseLayer(tf.keras.Model):
    def __init__(self, layers, dropout_keep_proba=0.9, activation=tf.nn.relu):
        super(DenseLayer, self).__init__()
        
        self.dense_layers = []
        self.dropout_keep_proba = dropout_keep_proba
        
        for i, layer_size in enumerate(layers):
            self.dense_layers.append(tf.keras.layers.Dense(layer_size, name='DenseLayer_' + str(i), use_bias=True, activation=tf.nn.relu))
    
    def call(self, logits):
        
        for layer in self.dense_layers:
            logits = layer(logits)
            logits = tf.nn.dropout(logits, keep_prob=self.dropout_keep_proba)

        return logits

# In[14]:


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
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

  hidden_size = output_layer.shape[-1].value
  n_ctx = input_ids.shape[-1].value
 
  transformer_outputs = bert_outputs['sequence_output']

  index_of_first_token = tf.argmax(segment_ids, axis=1)
  index_of_last_token = tf.argmax((1 - input_mask) * (1 - segment_ids), axis=1) - 1

  tf_range = tf.range(tf.shape(transformer_outputs)[0])
  tf_range = tf.cast(tf_range, tf.int64)
    
  index_of_first_token = tf.stack([tf_range, index_of_first_token], axis=1)
  index_of_last_token = tf.stack([tf_range, index_of_last_token], axis=1)
    
#   =========================================================================================================================
#   BIDIRECTIONAL 
  # take only last sentences
  last_sentences_transformer_outputs = transformer_outputs * tf.tile(tf.expand_dims(tf.cast(segment_ids, tf.float32), 2), [1, 1, hidden_size])
  # create a list of all LSTM cells we want
  num_layers = 1
  cells_fw = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(num_layers)]
  cells_bw = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(num_layers)]

  # we stack the cells together and create one big RNN cell
  cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
  cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
    
  inputs = tf.transpose(last_sentences_transformer_outputs, [1, 0, 2])
  inputs = tf.unstack(inputs, num=MAX_SEQ_LENGTH)
#   inputs = tf.reshape(inputs, [MAX_SEQ_LENGTH, None, hidden_size])

  outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(cell_fw,
                                                                             cell_bw, 
                                                                             inputs, 
                                                                             dtype=tf.float32)

  outputs = tf.stack(outputs)
  # outputs size [None, MAX_SEQ_LENGTH, hidden_size * 2]
  outputs = tf.transpose(outputs, [1, 0, 2])
    
  first_token_output = tf.gather_nd(outputs, index_of_first_token)
  last_token_output = tf.gather_nd(outputs, index_of_last_token)
  
  output_layer = tf.concat([first_token_output, last_token_output], 1)

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    logits = tf.layers.dense(output_layer, 512, use_bias=True, activation=tf.nn.sigmoid)
    logits = tf.nn.dropout(logits, keep_prob=0.9)
    logits = tf.layers.dense(logits, num_labels, use_bias=True)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# In[15]:


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
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


# In[16]:


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 50000
SAVE_SUMMARY_STEPS = 100000

OUTPUT_DIR = 'output_dir'
SAVE_RESULTS_DIR = 'results_predictions'

N_ESTIMATORS = 50

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / REPLICATION_FACTOR / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


assert REPLICATION_FACTOR == int(NUM_TRAIN_EPOCHS)

print('num_train_steps', num_train_steps)

# Skip this step to avoid disk quota
# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)


# In[17]:

def get_final_predictions(in_contexts, in_last_sentences):
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = y, label = 0) for x, y in zip(in_contexts, in_last_sentences)] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    predictions = [prediction['probabilities'] for prediction in predictions]

    return predictions

def combine_predictions(predictions):
    my_predictions = []

    i = 0
    while i < len(test):
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

# In[ ]:

import os

os.system('rm -rf results_predictions || true')
os.system('mkdir results_predictions')

true_labels_train = train_unshuffled['class'].values[::2] + 1
true_labels_val = test['class'].values[::2] + 1

for i in range(N_ESTIMATORS):
    os.system('rm -rf output_dir || true')
    
    train_features = shuffle(train_features)
    
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    predictions = get_final_predictions(test['story'].values, test['ending'].values)
    val_score = accuracy_score(true_labels_val, combine_predictions(predictions))
    np.savetxt(os.path.join("./" + SAVE_RESULTS_DIR, "predictions_test_" + str(val_score) + '_classifier_' + str(i) + '.csv'), predictions, delimiter=",")


# In[10]:


from os import listdir
from os.path import isfile, join
from scipy import stats
import numpy as np

SAVE_RESULTS_DIR = 'results_predictions'


files = [f for f in listdir(SAVE_RESULTS_DIR) if isfile(join(SAVE_RESULTS_DIR, f))]

true_labels_train = train_unshuffled['class'].values[::2] + 1
true_labels_test = test['class'].values[::2] + 1

classifiers = [int(file.split("_")[2].split(".")[0]) for file in files]
num_classifiers = np.max(classifiers)

predictions_train = []
predictions_test = []
for i in range(num_classifiers + 1):
    predictions_file_train = np.genfromtxt(os.path.join("./" + SAVE_RESULTS_DIR, 
                                                        "predictions_train_" + str(i) + '.csv'), delimiter=',')
    predictions_train.append(predictions_file_train)
    
    predictions_file_test = np.genfromtxt(os.path.join("./" + SAVE_RESULTS_DIR, 
                                                       "predictions_test_" + str(i) + '.csv'), delimiter=',')
    predictions_test.append(predictions_file_test)
    print('Classifier' + str(i))
    print(accuracy_score(true_labels_train, combine_predictions(predictions_file_train)))
    print(accuracy_score(true_labels_test, combine_predictions(predictions_file_test)))
    
def print_ensemble_predictions(predictions, true_labels):
    preds_mode = [combine_predictions(p) for p in predictions]
    preds_mode = np.array(preds_mode)
    preds_mode = stats.mode(preds_mode)[0][0]

    print('ENSEMBLE ACCURACY MODE')
    print(accuracy_score(true_labels, preds_mode))

    preds_prob = np.mean(predictions, axis=0)
    preds_prob = combine_predictions(preds_prob)

    print('ENSEMBLE ACCURACY PROB MEAN ON LOGS')
    print(accuracy_score(true_labels, preds_prob))


    preds_prob = np.log(np.mean(np.exp(predictions), axis=0))
    preds_prob = combine_predictions(preds_prob)

    print('ENSEMBLE ACCURACY PROB MEAN ON PROBS')
    print(accuracy_score(true_labels, preds_prob))
    print()
    
    
print_ensemble_predictions(predictions_train, true_labels_train)
print_ensemble_predictions(predictions_test, true_labels_test)

