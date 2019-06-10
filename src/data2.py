import re
import sys
import bert
import sklearn
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm 
from bert import tokenization

# use the BERT tokenizer
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# get bert tokenizer
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def fetch_data(train_file='train_stories.csv', valid_file='cloze_test_val__spring2016 - cloze_test_ALL_val.csv', fix_seed=False):
    """
    Load raw data.

    Parameters:
    -----------
    train_file : string 
        Training file name. 
    
    valid_file : string 
        Validation file name.

    fix_seed : boolean 
        Whether to fix the random seed. 

    Returns:
    --------
    A dictionary holding the following:

    train_stories : array-like, shape = (n_samples, 5)
        Training stories. 

    valid_stories : array-like, shape = (n_samples, 5)
        Validation stories.

    valid_labels : array-like, shape = (n_samples, )
        Validation labels. 

    """

    # Load raw data 
    train_stories = train_file.copy()

    valid_data = valid_file.copy()

    valid_stories = valid_data.drop('AnswerRightEnding', axis=1, inplace=False)
    valid_labels = valid_data['AnswerRightEnding'].values # to numpy array 

    # Fix the random seed
    if fix_seed:
        np.random.seed(13)

    # Training data
    train_stories = train_stories.drop('storyid', axis=1)
    train_stories = train_stories.drop('storytitle', axis=1)
    train_stories = train_stories.values # to numpy array 

    # Validation data 
    valid_stories = valid_stories.drop('InputStoryid', axis=1)
    valid_stories = valid_stories.values # to numpy array 

    return {'train': train_stories, 'valid': (valid_stories, valid_labels)}

def fetch_embedded_data(train_file='skip-thoughts-embbedings.npy', valid_file='skip-thoughts-embbedings_validation.npy'):

    path_to_embeddings = '/cluster/project/infk/courses/machine_perception_19/Sasglentamekaiedo/'

    train_stories = np.load(path_to_embeddings + train_file)
    valid_stories = np.load(path_to_embeddings + valid_file)

    valid_data = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv', index_col=False)
    valid_labels = valid_data['AnswerRightEnding']

    return {'train': train_stories, 'valid': (valid_stories, valid_labels)}

def random_ending(story_id, pos_stories):
    """
    Generate negative endings for the given story by randomly selecting endings from 
    different stories in the training set. 

    Parameters: 
    -----------
    story_id : int
        The index of the given story. 
    
    pos_stories : array-like, shape=(n_stories,5)
        The positive training stories.

    Returns: 
    --------
    neg_story : list 
        The negatively generated story 
    """

    pos_IDs = np.arange(pos_stories.shape[0])
    rand_ID = np.random.choice(np.delete(pos_IDs, story_id), replace=False)

    sampled_story = pos_stories[rand_ID, :].copy()
    rand_ending = sampled_story[-1]

    return rand_ending

def generate_negative_endings(pos_stories, method='random'):
    """
    Generate negative training examples. 

    Parameters:
    -----------
    pos_stories : pandas dataframe
        A dataframe with positive training stories. 

    method : string 
        Which method to use to generate negative endings. 

    Returns: 
    --------
    train_stories : pandas dataframe
        The input dataframe appended with negative training stories. 

    train_labels : pandas series 
        Training labels indicating whether the story ending is right (label = 1) or wrong (label = 0). 
    """
    N = pos_stories.shape[0]

    # Initialization 
    neg_endings = []

    for i in tqdm(range(N), desc='Negative Sampling'):     
        # Choose generating method
        if method == 'random':
            # Append negative stories 
            neg_ending = random_ending(i, pos_stories)
            neg_endings.append(neg_ending)

    neg_endings = np.array(neg_endings)
    neg_endings = neg_endings[:, np.newaxis]

    # Append to positive stories 
    train_stories = np.hstack((pos_stories, neg_endings))

    # Training labels 
    pos_labels = np.ones(N, dtype=np.int32)
    neg_labels = np.zeros(N, dtype=np.int32)
    train_labels = np.hstack((pos_labels[:, np.newaxis], neg_labels[:, np.newaxis]))

    return train_stories, train_labels

def shuffle_endings(corpus, labels):
    
    N = corpus.shape[0]
    context = corpus[:,:4]
    endings = corpus[:,4:]

    shuffled_labels = []
    shuffled_endings = []

    for i in tqdm(range(N), desc='Shuffling Endings'):
        # Text 
        e = endings[i,:]
        e_ = e.copy()

        # Labels 
        l = labels[i,:]
        l_ = l.copy()

        swap = np.random.choice([True, False])

        if swap:
            e_[0] = e[1]
            e_[1] = e[0]

            l_[0] = l[1]
            l_[1] = l[0]

        shuffled_labels.append(l_)
        shuffled_endings.append(e_)

    shuffled_labels = np.array(shuffled_labels)
    shuffled_endings = np.array(shuffled_endings)

    shuffled_corpus = np.hstack((context, shuffled_endings))

    return shuffled_corpus, shuffled_labels

def construct_vocab(corpus, base_vocab={'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}):
    """
    Associate each word in the vocabulary with a (unique) ID. 

    Parameters:
    -----------
    corpus : array-like, shape=(no_stories, story_len) 
        Array with stories. 

    base_vocab: dict
        Initialization for the vocabulary. 

    Returns: 
    --------
    vocab : dict 
        The vocabulary dictionary where keys correspond to (unique) words in the vocabulary and the values correspond to the unique ID of the word.

    max_len : int
        The maximum sentence length from the corpus. 
    """
    max_len = 0
    counter = collections.Counter()

    tokenizer = create_tokenizer_from_hub_module()

    for story in tqdm(corpus, desc='Constructing Vocabulary',position=0):
        for sentence in story: 
            # Tokenize sentence 
            tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]
            counter.update(tokens)

            # Find the maximum sentence length 
            if len(tokens) > max_len:
                max_len = len(tokens)

    most_common = counter.most_common() 

    # Initialize the vocabulary 
    vocab = dict(base_vocab)

    # Associate each word in the vocabulary with a unique ID number
    ID = len(base_vocab)

    for token, _ in most_common:
        vocab[token] = ID
        ID += 1

    # Include <bos> and <eos> tokens
    max_len = max_len + 2

    inverse_vocab = {v: k for k, v in vocab.items()}

    return vocab, inverse_vocab, max_len

def encode_text(corpus, max_len, vocab): 
    """
    Encode words in the text in terms of their ID in the vocabulary. Sentences that are longer than 30 tokens (including <bos> and <eos> are ignored).

    Parameters: 
    -----------
    corpus : list 
        Each entry in the list corresponds to a sentence in the corpus

    max_len : int
        The maximum sentence length.

    vocab : dict
        The vocabulary dictionary

    Returns:
    --------
    data : array-like, shape (n_sentences, sentence_len)
        Each row corresponds to a sentence. Entries in a row are integers and correspond to the vocabulary word ID.
    """

    tokenizer = create_tokenizer_from_hub_module()

    no_stories = corpus.shape[0]

    context_corpus = corpus[:,:4]
    endings_corpus = corpus[:,4:]

    # Initialize the data matrix
    context = np.full(shape=(no_stories, 4*max_len), fill_value=vocab['<pad>'], dtype=int)
    endings = np.full(shape=(no_stories, 2*max_len), fill_value=vocab['<pad>'], dtype=int)
    
    # Encode story context 
    story_ID = 0 
    long_context = 0

    for story in tqdm(context_corpus, desc='Encoding Context',position=0):
        # Reset pointer 
        token_ID = 0
        
        for sentence in story: 
            # Beggining of sentence  
            context[story_ID, token_ID] = vocab['<bos>'] 
            token_ID += 1

            # Tokenize the sentence 
            tokens = tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]

            # For long test sentences
            if len(tokens) > (max_len - 2):
                tokens = tokens[:(max_len - 2)] 
                long_context += 1

            for token in tokens: 
                if token in vocab.keys():
                    context[story_ID, token_ID] = vocab[token] # Within-vocabulary
                else:
                    context[story_ID, token_ID] = vocab['<unk>'] # Out-of-vocabulary

                token_ID += 1

            # End of sentence
            context[story_ID, token_ID] = vocab['<eos>'] 
            token_ID += 1

        story_ID += 1

    print('%i longer sentences were found in context.' %long_context)

    # Encode endings 
    story_ID = 0 
    long_ending = 0 

    for story in tqdm(endings_corpus, desc='Encoding Endings',position=0):
        # Reset pointer 
        token_ID = 0
        
        for sentence in story: 
            # Beggining of sentence  
            endings[story_ID, token_ID] = vocab['<bos>'] 
            token_ID += 1

            # Tokenize the sentence 
            tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]

            # For long test sentences
            if len(tokens) > (max_len - 2):
                tokens = tokens[:(max_len - 2)] 
                long_ending += 1

            for token in tokens: 
                if token in vocab.keys():
                    endings[story_ID, token_ID] = vocab[token] # Within-vocabulary
                else:
                    endings[story_ID, token_ID] = vocab['<unk>'] # Out-of-vocabulary

                token_ID += 1

            # End of sentence
            endings[story_ID, token_ID] = vocab['<eos>'] 
            token_ID += 1

            # Move the pointer 
            token_ID = int(endings.shape[1]/2)

        story_ID += 1

    print('%i longer sentences were found in endings.' %long_ending)

    # stories = np.hstack((context, endings))

    return np.concatenate((context, endings),axis=1)

def decode_sentence(tokens, inverse_vocab):
    
    return ' '.join(list(map(inverse_vocab.get, tokens))[1:])
