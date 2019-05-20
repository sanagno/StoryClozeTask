import sys
import sklearn

import numpy as np
import pandas as pd

# Custom dependencies 
import data
import model_2

if __name__ == '__main__': 

    dataloader = data.fetch_embedded_data()

    pos_stories = dataloader['train']
    valid_stories, valid_labels = dataloader['valid']

    # valid_labels = valid_labels[:, np.newaxis]

    train_stories, train_labels = data.generate_negative_endings(pos_stories)    

    print('Training stories: ', train_stories.shape)
    print('Training labels: ', train_labels.shape)

    embedding_dim = train_stories.shape[2]

    rnn_baseline = model_2.RNNBaseline(cell='gru', input_size=embedding_dim)

    rnn_baseline.train(train_stories, train_labels, x_val=valid_stories, y_val=valid_labels, epochs=15, optimizer='adam', learning_rate=0.03)

    print('Validation Accuracy: ', rnn_baseline.score(valid_stories, valid_labels))



