import sys
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm 

def fetch_data(train_file='train_stories.csv', valid_file='cloze_test_val__spring2016 - cloze_test_ALL_val.csv', fix_seed=False):
    """
    Load raw data

    Parameters:
    -----------
    train_file : string 
        Training file name 
    
    valid_file : string 
        Validation file name 

    fix_seed : boolean 
        Whether to fix the random seed 

    Returns:
    --------
    A dictionary holding the following

    train_stories : array-like, shape = (n_samples, 5)
        Training stories 

    valid_stories : pandas dataframe 
        Validation stories

    valid_labels : pandas series 
        Validation labels 

    """

    # TODO: Depending on testing pipeline change valid_stories from dataframe to numpy array. 

    # Load raw data 
    train_stories = pd.read_csv('data/%s' %train_file, index_col=False)

    valid_data = pd.read_csv('data/%s' %valid_file, index_col=False)

    valid_stories = valid_data.drop('AnswerRightEnding', axis=1)
    valid_labels = valid_data['AnswerRightEnding']

    # Fix the random seed
    if fix_seed:
        np.random.seed(13)

    # Training data
    train_stories = train_stories.drop('storyid', axis=1)
    train_stories = train_stories.drop('storytitle', axis=1)
    train_stories = train_stories.values 

    # Validation data 
    valid_stories = valid_stories.drop('InputStoryid', axis=1)

    return {'train': train_stories, 'valid': (valid_stories, valid_labels)}

def fetch_embedded_data(train_file='skip-thoughts-embbedings.npy', valid_file='skip-thoughts-embbedings_validation.npy'):

    path_to_embeddings = '/cluster/project/infk/courses/machine_perception_19/Sasglentamekaiedo/'

    train_stories = np.load(path_to_embeddings + train_file)
    valid_stories = np.load(path_to_embeddings + valid_file)

    valid_data = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv', index_col=False)
    valid_labels = valid_data['AnswerRightEnding']

    return {'train': train_stories, 'valid': (valid_stories, valid_labels)}

def random_ending(story_id, positive_stories):
    """
    Generate negative endings for the given story by randomly selecting endings from 
    different stories in the training set. 

    Parameters: 
    -----------
    story_id : int
        The index of the given story. 
    
    positive_stories : array-like, shape=(n_stories,5)
        The positive training stories.

    Returns: 
    --------
    neg_story_lst : array-like, shape=(n,5)
    The negatively generated stories
    """

    # Number of negative samples to be generated
    n = np.random.randint(low=1, high=4)

    pos_IDs = np.arange(positive_stories.shape[0])

    # Draw a set of positive IDs uniformly at random
    drawn_IDs = np.random.choice(np.delete(pos_IDs, story_id), size=n, replace=False)

    neg_story_lst = []

    for i in drawn_IDs:
        # Sample the given story 
        neg_story = positive_stories[story_id,:,:] 
        # Replace the fifth sentence with the negative ending 
        neg_story[-1,:] = positive_stories[i,-1,:]
        # At to the list 
        neg_story_lst.append(neg_story)

    return np.array(neg_story_lst)

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
        Training labels indicating whether the story ending is right or wrong. 
        The following is used, 0 : Wrong ending, 1 : Right ending. 
    """
    N_pos = pos_stories.shape[0]

    # Initialization 
    neg_stories = []
    train_labels = np.ones(N_pos, dtype=np.int32)

    for ID in range(N_pos):     
        # Choose generating method
        if method == 'random':
            # Append negative stories 
            neg_stories.append(random_ending(ID, pos_stories))

            # Append the negative stories and labels to the training set
            # train_stories = np.vstack((train_stories, neg_stories))
            # train_labels  = np.append(train_labels, np.zeros(neg_stories.shape[0], dtype=np.int8))

    neg_stories = np.concatenate(neg_stories, axis=0)
    N_neg = neg_stories.shape[0]

    # Append to positive stories 
    train_stories = np.vstack((pos_stories, neg_stories))
    train_labels  = np.append(train_labels, np.zeros(N_neg, dtype=np.int32))
    # train_labels  = train_labels[:, np.newaxis]   

    train_stories, train_labels = sklearn.utils.shuffle(train_stories, train_labels)

    return (train_stories, train_labels)

def encode_stories(stories):

    N_stories = stories.shape[0]

    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    # Initialize output 
    embedded_stories = []

    for story_ID in tqdm(range(N_stories)):
        embedded_stories.append(encoder.encode(stories[story_ID,:].tolist()))

    return embedded_stories

