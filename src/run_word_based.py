import sys
import numpy as np
import pandas as pd

from tqdm import tqdm 

# Custom dependencies 
import data
from word_based_model import WordBasedClassifier

if __name__ == '__main__': 

    # Load data
    dataloader = data.fetch_data()

    train_pos_stories = dataloader['train']
    valid_stories, valid_labels = dataloader['valid']
    test_stories,  test_labels  = dataloader['test']

    # Load negative endings 
    neg_endings = np.load('incorrect_endings/incorrect_endings_v2_argmax.npy')
    neg_endings = neg_endings[:, np.newaxis]

    # Append to positive stories 
    train_stories = np.hstack((train_pos_stories, neg_endings))

    # Training labels 
    train_labels = np.ones(train_stories.shape[0], dtype=np.int32)

    # Construct vocabulary 
    vocab, _, max_len = data.construct_vocab(train_stories)

    # Model 
    model = WordBasedClassifier(vocab, sentence_len=max_len)

    # Get encoded training data 
    encoded_train_stories, train_labels = model.get_train_data(valid_stories, valid_labels, shuffle=True)

    # Get encoded validation data 
    encoded_valid_stories = model.get_test_data(valid_stories)
    encoded_test_stories  = model.get_test_data(test_stories)

    # Train the model 
    model.fit(encoded_train_stories, train_labels, encoded_test_stories, test_labels, epochs=10, learning_rate=5e-3)

    # print('Validation Accuracy: ', model.score(encoded_valid_stories, valid_labels))

    print('Test Accuracy: ', model.score(encoded_test_stories, test_labels))


    


