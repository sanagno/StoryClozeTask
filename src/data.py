import collections
import bert
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random


BASE_VOCAB = {'<unk>': 0, '<pad>': 1}

# length of one sentence
PADDED_LENGTH = 20

# use the BERT tokenizer
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# get bert tokenizer
def _create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def sample_negatives(all_data, data_batch, classes_batch, negative_sampling):
    new_classes = []
    new_data_batch = []
    for i, data in enumerate(data_batch):
        new_data_batch.append(data)
        new_classes.append(classes_batch[i])

        for _ in range(negative_sampling):
            new_data_batch.append(
                np.concatenate((data[:4], [random.choice(all_data[:, 4])]), axis=0))
            # negative class always
            new_classes.append(1)
    return np.array(new_data_batch), np.array(new_classes, dtype=np.int32)


class DataProcessing:
    _vocab = None
    _inverse_vocab = None
    train_data = None
    train_classes = None
    val_data = None
    val_classes = None
    test_data = None
    test_classes = None

    def __init__(self, vocabulary_size=50000):
        self.vocabulary_size = vocabulary_size
        self.tokenizer = _create_tokenizer_from_hub_module()

    def read_data_train(self, path):
        """
        Parameters
        ----------
        path:   Where to read the data from

        Returns
        -------
        Tuple of story lowercase and zeros denoting that all ending correspond to true endings
        """
        train_data_pd = pd.read_csv(path, header='infer')
        train_data = train_data_pd[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].values
        train_data = np.array([[xii.lower() for xii in xi] for xi in train_data])

        return shuffle(train_data), np.zeros(len(train_data))

    def read_data_val(self, path, test=False):
        """
        Parameters
        ----------
        path:   Where to read the data from
        test:   Whether it is the validation of test set

        Returns
        -------
        Tuple of story lowercase and array of values 0s and 1s. 0s corresponds to true endings and 1s to false ones
        """
        data_pd = pd.read_csv(path, header='infer')

        # create dataset

        data = list()
        classes = list()
        correct_answers = data_pd['AnswerRightEnding'].values

        def lower_story(story):
            return np.array([s.lower() for s in story])

        for i, x in enumerate(data_pd.values):
            data.append(lower_story(x[1:6]))
            data.append(lower_story(np.append(x[1:5], [x[6]], axis=0)))

            if correct_answers[i] == 1:
                classes.append(0)
                classes.append(1)
            else:
                classes.append(1)
                classes.append(0)

        return np.array(data), np.array(classes)

    def construct_vocab(self, data=None):
        """
        Parameters
        ----------
        data:       Array of list of strings. Each List corresponds to one story

        -------
        Creates vocabulary based on the stories provided
        """
        if data is None:
            data = self.train_data
            if data is None:
                raise Exception("Either specify data or training data.")

        tokenized_sentences = []
        for sentence in data:
            # tokens that start with a character '#' do not have
            tokens = [x for x in self.tokenizer.tokenize(" ".join(sentence)) if not x.startswith('#')]
            tokenized_sentences.append(tokens)

        self._construct_vocab_from_tokens(tokenized_sentences)

    def _construct_vocab_from_tokens(self, tokenized_sentences, base_vocab=BASE_VOCAB):
        counter = collections.Counter()

        for tokenized_sentence in tokenized_sentences:
            counter.update(tokenized_sentence)

        print('Total distinct tokens found:', len(counter))
        # Keep the vocab_size - (base_vocab size) most common words from the corpus
        most_common = counter.most_common(self.vocabulary_size - len(base_vocab))

        # Initialize the vocabulary
        vocab = dict(base_vocab)

        # Associate each word in the vocabulary with a unique ID number
        token_id = len(base_vocab)

        for token, _ in most_common:
            vocab[token] = token_id
            token_id += 1

        # token to token_id mappings
        inverse_vocab = {v: k for k, v in vocab.items()}

        self._vocab = vocab
        self._inverse_vocab = inverse_vocab

    def _encode_text(self, stories, vocab, padded_length):
        encoded_stories = []

        # Fill-in the data matrix
        for story in stories:
            tokenized_story = self.tokenizer.tokenize(" ".join(story))

            encoded_story = []

            length = 0

            for token in tokenized_story:
                if length >= padded_length:
                    break

                if token in vocab:
                    encoded_story.append(vocab[token])  # Within-vocabulary
                    length += 1

            while length < padded_length:
                encoded_story.append(vocab['<pad>'])
                length += 1

            encoded_stories.append(encoded_story)

        return np.array(encoded_stories, dtype=np.int32)

    def encode_story(self, stories):
        return self._encode_text(stories[:, 0:4], self.vocab, padded_length=PADDED_LENGTH * 4), \
               self._encode_text(stories[:, 4][:, np.newaxis], self.vocab, padded_length=PADDED_LENGTH)

    def decode_sentence(self, sentence):
        return ' '.join(list(map(self.inverse_vocab.get, sentence)))

    def generator(self, mode, batch_size=64, negative_sampling=0):
        """
        Parameters
        ----------
        mode                Set to use. O -> train, 1 -> validation, 2 -> test
        batch_size          Batch size
        negative_sampling   How many negatives per positive sentence

        Returns
        -------
        Tuple of encoding of first 4 sentences, last sentence and class.
        Encoding is based on the vocab previously constructed
        """
        if self._vocab is None:
            raise Exception("Vocabulary has not been specified yet.")

        if mode == 0:
            data = self.train_data
            classes = self.train_classes
        elif mode == 1:
            data = self.val_data
            classes = self.val_classes
        elif mode == 2:
            data = self.test_data
            classes = self.test_classes
        else:
            raise Exception("Mode " + str(mode) + " not supported.")

        if data is None or classes is None:
            raise Exception("Not available data for mode " + str(mode))


        if mode == 'train' and negative_sampling > 0:
            batch_size /= (negative_sampling + 1)
            if batch_size != int(batch_size):
                raise Exception('Batch size should be an integer. Please change negative sampling rate')

            batch_size = int(batch_size)

        # repeat
        while (True):
            if mode == 0:
                data, classes = shuffle(data, classes)

            length = len(data)
            for ndx in range(0, length, batch_size):
                data_batch = data[ndx: min(ndx + batch_size, length)]
                classes_batch = classes[ndx: min(ndx + batch_size, length)]

                if negative_sampling > 0:
                    data_batch, classes_batch = sample_negatives(data, data_batch, classes_batch, negative_sampling)

                context_batch, last_sentence_batch = self.encode_story(data_batch)
                yield context_batch, last_sentence_batch, classes_batch

    def create_dataset(self, mode, batch_size, negative_sampling):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.int32, tf.int32, tf.int32),
                                                 output_shapes=(tf.TensorShape([None, PADDED_LENGTH * 4]),
                                                                tf.TensorShape([None, PADDED_LENGTH]),
                                                                tf.TensorShape([None])),
                                                 args=([mode, batch_size, negative_sampling]))

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    @property
    def vocab(self):
        return self._vocab

    @property
    def inverse_vocab(self):
        return self._inverse_vocab

    @property
    def num_samples_training(self):
        return len(self.train_data)

    @property
    def num_samples_val(self):
        return len(self.val_data)

    @property
    def num_samples_test(self):
        return len(self.test_data)
