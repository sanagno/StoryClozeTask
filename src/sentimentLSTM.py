#!/bin/env python

# The LSTM model for sentiment analysis as described in paper:
# Incorporating Structured Commonsense Knowledge in Story Completion
# https://arxiv.org/pdf/1811.00625.pdf

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, Dense
import keras.backend as K
from keras.models import Sequential
from nltk import sent_tokenize
import pandas as pd
import numpy as np

# VAL_SET = 'test_for_report-stories_labels.csv' # Which is actually test set
TEST_SET = 'test_for_report-stories_labels.csv' # Which is actually test set
ROC_VAL_SET = 'train_stories.csv'
# DATA_DIR = '../data'
DATA_DIR = 'data/ROCStories'

from model import NLUModel
analyzer = SentimentIntensityAnalyzer()

def polarity_no_compound(sentence):
    '''
    Convert a sentence to a np.array containing the polarities
    '''
    polarity_dict = analyzer.polarity_scores(sentence)
    polarity_dict.pop('compound')
    values = polarity_dict.values()
    return np.array(list(values))

def polarity_full(sentence):
    '''
    Convert a sentence to a np.array containing the polarities
    '''
    polarity_dict = analyzer.polarity_scores(sentence)
    values = polarity_dict.values()
    return np.array(list(values))

class SentimentLSTM(NLUModel):

    def __init__(self):
        super(SentimentLSTM, self).__init__('SentimentLSTM')
        self.__build()

    def __build(self):
        self.model = Sequential()
        self.model.add(LSTM(3, input_shape=(4,3)))
        self.model.add(Dropout(0.1))
        polarity_layer = self.model.add(Dense(3, activation='softmax'))

        # Optimize the model so the polarity of the layer is as close as the correct polarity
        def cosine_similarity_loss(layer):
            # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
            def loss(y_true,y_pred):
                pred_norm = y_pred/K.square(K.batch_dot(y_pred,y_pred, axes=[1,1]))
                true_norm = y_true/K.square(K.batch_dot(y_true,y_true, axes=[1,1]))
                cosine = K.batch_dot(pred_norm, true_norm, axes=[1,1])
                cosine = K.sum(cosine)
                return -cosine
            # Return a function
            return loss

        self.model.compile(loss=cosine_similarity_loss(polarity_layer),
                      optimizer='adam')

    def fit(self, X, y, epochs=10, batch_size=16):

        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)

    def predict(self, X):

        X_test = X[0]
        answer1 = X[1]
        answer2 = X[2]
        predictions = self.model.predict(X_test[:,:,[0,1,2]])
        compound_mean = np.mean(X_test[:,:,3], axis=-1)

        predictions_with_compound = np.zeros((predictions.shape[0], 4))
        predictions_with_compound[:,[0,1,2]] = predictions
        predictions_with_compound[:,3] = compound_mean

        final_predictions = np.zeros((predictions.shape[0],))

        for i in range(predictions.shape[0]):
            if cosine_similarity(predictions_with_compound[i].reshape(1,-1), answer1[i].reshape(1,-1)) \
             < cosine_similarity(predictions_with_compound[i].reshape(1,-1), answer2[i].reshape(1,-1)):
                y = 2
            else:
                y = 1
            final_predictions[i] = y

        return final_predictions

    def prepare_test_dataset(self, df):
        '''
        Convert test set to polarity set.
        '''
        X = []
        Y = []
        input_sentences = ['InputSentence%d'%i for i in range(1,5)]
        first_sentences = df[input_sentences]
        beginning_polarities = first_sentences.apply(lambda x: np.stack(x.apply(polarity_full).values), axis=1)
        ending_polarity1 = np.stack(df['RandomFifthSentenceQuiz1'].apply(polarity_full).values, axis=0)
        ending_polarity2 = np.stack(df['RandomFifthSentenceQuiz2'].apply(polarity_full).values, axis=0)
        correct = df['AnswerRightEnding'].values
        X = np.stack(beginning_polarities.values,axis=0)
        return X, ending_polarity1, ending_polarity2, correct

    def prepare_roc_dataset(self, df):
        input_sentences = ['sentence%d'%i for i in range(1,5)]
        first_sentences = df[input_sentences]
        beginning_polarities = first_sentences.apply(lambda x: np.stack(x.apply(polarity_no_compound).values), axis=1)
        ending = df.drop(columns = input_sentences)
        ending_polarity = ending['sentence5'].apply(polarity_no_compound)
        X = np.stack(beginning_polarities.values,axis=0)
        Y = np.stack(ending_polarity.values)
        return X,Y

    def get_train_data(self, nrows=None):
        roc_df = pd.read_csv(DATA_DIR + '/' + ROC_VAL_SET, nrows=nrows).drop(columns=['storyid','storytitle'])
        X, y = self.prepare_roc_dataset(roc_df)
        return X,y

    def get_test_data(self):
        val_df = pd.read_csv(DATA_DIR + '/' + TEST_SET).drop(columns=['InputStoryid'])
        X_test, answer1, answer2, y_test = self.prepare_test_dataset(val_df)
        return [X_test, answer1, answer2], y_test

    def evaluate(self, true_y, pred_y):
        return accuracy_score(true_y, pred_y)




if __name__  == "__main__":

    sent_model = SentimentLSTM()
    trX, trY = sent_model.get_train_data(nrows=100)
    teX, teY = sent_model.get_test_data()
    sent_model.fit(trX, trY, epochs=1)
    y_pred = sent_model.predict(teX)
    score = sent_model.evaluate(teY, y_pred)
    print('SentimentLSTM score (on test_set):', score)


