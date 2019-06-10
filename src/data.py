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

TRAIN_FILE = 'train_stories.csv'
VALID_FILE = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
TEST_FILE  = 'test_for_report-stories_labels.csv'

# use the BERT tokenizer
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """
    Use the tokenizer used by BERT. 

    Get the vocab file and casing info from the Hub module
    """
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def fetch_data():
    """
    Load raw data.

    Parameters:
    -----------
    train_file : string 
        Training file name. 
    
    valid_file : string 
        Validation file name.

    Returns:
    --------
    A dictionary holding the following:

    train_stories : array-like, shape = (n_train, 5)
        Each row corresponds to a story and each column to a sentence. The first four 
        columns correspond to the story context and the final column to the ending. 

    valid_stories : array-like, shape = (n_valid, 6)
        Each row corresponds to a story and each column to a sentence. The first four 
        columns correspond to the story context and the final two columns to the endings. 

    valid_labels : array-like, shape = (n_valid, )
        Validation labels indicating which ending is correct (1 or 2). 
    """
    # Load raw data 
    train_stories = pd.read_csv('data/%s' %TRAIN_FILE, index_col=False)

    valid_data = pd.read_csv('data/%s' %VALID_FILE, index_col=False)
    test_data  = pd.read_csv('data/%s' %TEST_FILE,  index_col=False)
    
    # Training stories
    train_stories = train_stories.drop('storyid', axis=1)
    train_stories = train_stories.drop('storytitle', axis=1)
    train_stories = train_stories.values # to numpy array 

    # Validation stories 
    valid_stories = valid_data.drop('AnswerRightEnding', axis=1, inplace=False)
    valid_stories = valid_stories.drop('InputStoryid', axis=1)
    valid_stories = valid_stories.values # to numpy array 

    # Validation data
    valid_labels  = valid_data['AnswerRightEnding'].values # to numpy array 

    # Test stories 
    test_stories = test_data.drop('AnswerRightEnding', axis=1, inplace=False)
    test_stories = test_stories.drop('InputStoryid', axis=1)
    test_stories = test_stories.values # to numpy array 

    # Validation data
    test_labels  = test_data['AnswerRightEnding'].values # to numpy array 

    return {'train': train_stories, 'valid': (valid_stories, valid_labels), 'test': (test_stories, test_labels)}

def shuffle_endings(corpus, labels):
    """
    Shuffle the two ending alternatives. 

    Parameters: 
    -----------
    corpus : array-like, shape=(n_samples, 6)
        In each row, the first 4 entries correspond to the story context and the last two to the two ending alternatives. 

    labels : array-like, shape=(n_samples, )
        Labels for the stories. They take values 1 or 2 indicating which story ending is the right one. 

    Returns: 
    --------
    shuffled_corpus : array-like, shape=(n_samples, 6)
        In each row, the first 4 entries correspond to the story context and the last two to the two shuffled ending alternatives. 

    shuffled_labels : array-like. shape=(n_samples, )
        Shuffled labels for the stories. They take values 1 or 2 indicating which story ending is the right one (after shuffling). 
    """
    N = corpus.shape[0]
    context = corpus[:,:4]
    endings = corpus[:,4:]

    shuffled_endings = []
    shuffled_labels  = labels.copy()

    for i in range(N):

        # Text 
        e = endings[i,:]
        e_ = e.copy()

        # Whether to shuffle the story endings or not. 
        swap = np.random.choice([True, False])

        if swap:
            e_[0] = e[1]
            e_[1] = e[0]

            shuffled_labels[i] = 2

        shuffled_endings.append(e_)

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
        The vocabulary dictionary where keys correspond to (unique) words in the vocabulary and the values 
        correspond to the unique ID of the word.
    
    inverse_vocab : dict 
        The inverse of the vocabulary where keys correspond to (unique) IDs in the vocabulary and the values 
        correspond to the associated word. 

    max_len : int
        The maximum sentence length from the corpus. 
    """
    max_len = 0
    counter = collections.Counter()

    tokenizer = create_tokenizer_from_hub_module()

    for story in tqdm(corpus, desc='Constructing Vocabulary', position=0):
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

    # Inverse vocabulary
    inverse_vocab = {v: k for k, v in vocab.items()}

    return vocab, inverse_vocab, max_len

def encode_text(corpus, max_len, vocab): 
    """
    Encode words in the text in terms of their ID in the vocabulary. Sentences that are longer than max_len 
    tokens (including <bos> and <eos>) are ignored. Padding is applied to bring all sentences to the same length. 

    Parameters: 
    -----------
    corpus : array-like 
        Each entry in the list corresponds to a sentence in the corpus

    max_len : int
        The maximum sentence length including <bos> and <eos> tokens.

    vocab : dict
        The vocabulary dictionary.

    Returns:
    --------
    context : array-like, shape=(n_stories, 4*max_len)
        Entries in a row are integers and correspond to the vocabulary word ID.

    endings : array-like, shape=(n_stories, 2*max_len)
        Entries in a row are integers and correspond to the vocabulary word ID.
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

    for story in tqdm(context_corpus, desc='Encoding Context', position=0):
        # Reset pointer 
        token_ID = 0
        
        for sentence in story: 
            # Beggining of sentence  
            context[story_ID, token_ID] = vocab['<bos>'] 
            token_ID += 1

            # Tokenize the sentence 
            tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]

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

    for story in tqdm(endings_corpus, desc='Encoding Endings', position=0):
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

    return context, endings

def encode_train_text_for_conditional_generation(corpus, max_len, vocab):
    """
    Encode the data to be used for conditional ending generation using the language model. 

    Parameters: 
    -----------
    corpus : array-like, shape=(n_stories, 5)
        The ROCStories with only one ending (the true one). 

    max_len : int 
        Maximum sentence length including <bos> and <eos> tokens. 

    vocab : dict 
        The vocabulary dictionary.

    Returns: 
    --------
    context_nopad : list   
        Each entry in the list corresponds to an encoded story with no <pad> symbols. 

    endings_nopad : list 
        Each entry in the list corresponds to an encoded (true) ending with no <pad> symbols. 
    """
    tokenizer = create_tokenizer_from_hub_module()

    no_stories = corpus.shape[0]

    context_corpus = corpus[:,:4] 
    endings_corpus = corpus[:, 4] # Single ending!

    context = np.full(shape=(no_stories, 4*max_len), fill_value=vocab['<pad>'], dtype=int)
    endings = np.full(shape=(no_stories, max_len), fill_value=vocab['<pad>'], dtype=int)

    # Output array 
    context_nopad = []
    endings_nopad = []

    # Encode story context 
    story_ID = 0 
    long_context = 0

    for story in tqdm(context_corpus, desc='Encoding Context (No Padding)', position=0):
        # Reset pointer 
        token_ID = 0
        
        for sentence in story: 
            # Beggining of sentence  
            context[story_ID, token_ID] = vocab['<bos>'] 
            token_ID += 1

            # Tokenize the sentence 
            tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]

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

        # Append to output without pads
        context_nopad.append(context[story_ID, :token_ID])

        story_ID += 1

    print('%i longer sentences were found in endings.' %long_context)

    # Encode ending
    story_ID = 0 
    long_ending = 0 

    for sentence in tqdm(endings_corpus, desc='Encoding Endings', position=0):
        # Reset pointer 
        token_ID = 0

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

        endings_nopad.append(endings[story_ID, :token_ID])

        story_ID += 1

    return context_nopad, endings_nopad

def encode_valid_text_for_fine_tunning(corpus, labels, max_len, vocab):
    """
    Encodes validation data for finetuning purposes during training of the language model. That is, the function encodes 
    the story context followed by the wrong ending. Padding is applied after the end of the ending sentence. 

    Parameters: 
    -----------
    corpus : array-like, shape=(n_stories, 6)
        The validation stories. 

    labels : array-like, shape=(n_stories, )
        The labels for the given stories. 

    max_len : int 
        Maximum sentence length including <bos> and <eos> tokens. 

    vocab : dict 
        The vocabulary dictionary.

    Returns: 
    --------
    stories : array-like, shape=(n_stories, 5*max_len)
        In each row, the first four sentences correspond to the encoded context and the last sentence to the wrong ending. The 
        rest of the tokens are filled with <pad>.
    """
    tokenizer = create_tokenizer_from_hub_module()

    no_stories = corpus.shape[0]

    context_corpus = corpus[:,:4]
    endings_corpus = corpus[:,4:]

    # Initialize the data matrix
    stories = np.full(shape=(no_stories, 5*max_len), fill_value=vocab['<pad>'], dtype=int)

    for story_ID in tqdm(range(no_stories), desc='Encoding Stories for Finetunning', position=0):
        # Reset pointer 
        token_ID = 0

        # Slice the current story context
        story_context = context_corpus[story_ID, :]

        # Encode the context of the story 
        for sentence in story_context: 
            # Beggining of sentence 
            stories[story_ID, token_ID] = vocab['<bos>']
            token_ID += 1

            # Tokenize the sentence 
            tokens = [x for x in tokenizer.tokenize(sentence.lower()) if not x.startswith('#')]

            # For long test sentences
            if len(tokens) > (max_len - 2):
                tokens = tokens[:(max_len - 2)] 

            for token in tokens: 
                if token in vocab.keys():
                    stories[story_ID, token_ID] = vocab[token] # Within-vocabulary
                else:
                    stories[story_ID, token_ID] = vocab['<unk>'] # Out-of-vocabulary

                token_ID += 1

            # End of sentence
            stories[story_ID, token_ID] = vocab['<eos>'] 
            token_ID += 1

        # Continue by encoding the wrong ending 
        wrong_labels = np.zeros(len(labels), dtype=int)
        wrong_labels[labels==1] = 2
        wrong_labels[labels==2] = 1

        wrong_ending = endings_corpus[story_ID, (wrong_labels[story_ID] - 1)] 

        # Beggining of ending sentence
        stories[story_ID, token_ID] = vocab['<bos>']
        token_ID += 1

        tokens = [x for x in tokenizer.tokenize(wrong_ending.lower()) if not x.startswith('#')]

        # For long test sentences
        if len(tokens) > (max_len - 2):
            tokens = tokens[:(max_len - 2)] 

        for token in tokens: 
            if token in vocab.keys():
                stories[story_ID, token_ID] = vocab[token] # Within-vocabulary
            else:
                stories[story_ID, token_ID] = vocab['<unk>'] # Out-of-vocabulary

            token_ID += 1

        # End of sentence
        stories[story_ID, token_ID] = vocab['<eos>'] 
        token_ID += 1

    return stories

def decode_sentence(tokens, inverse_vocab):

    return ' '.join(list(map(inverse_vocab.get, tokens))[1:])
