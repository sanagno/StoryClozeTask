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

VAL_SET = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
ROC_VAL_SET = 'ROCStories__spring2016 - ROCStories_spring2016.csv'
TEST_SET = 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
DATA_DIR = 'dataset'
ENCODER_PATH = 'finetune-transformer-lm-master/model/encoder_bpe_40000.json'
BPE_PATH = 'finetune-transformer-lm-master/model/vocab_40000.bpe'
n_ctx = 512

seed = 42

analyzer = SentimentIntensityAnalyzer()
roc_df = pd.read_csv(DATA_DIR + '/' + ROC_VAL_SET).drop(columns=['storyid','storytitle'])

def prepare_test_dataset(df):

    X = []
    Y = []
    def polarity(sentence):
        values = analyzer.polarity_scores(sentence).values()
        return np.array(list(values))

    input_sentences = ['InputSentence%d'%i for i in range(1,5)]
    first_sentences = df[input_sentences]
    beginning_polarities = first_sentences.apply(lambda x: np.stack(x.apply(polarity).values), axis=1)
    ending_polarity1 = np.stack(df['RandomFifthSentenceQuiz1'].apply(polarity).values, axis=0)
    ending_polarity2 = np.stack(df['RandomFifthSentenceQuiz2'].apply(polarity).values, axis=0)
    correct = df['AnswerRightEnding'].values
    X = np.stack(beginning_polarities.values,axis=0)
    return X, ending_polarity1, ending_polarity2, correct

def prepare_roc_dataset(df):
    def polarity(sentence):
        values = analyzer.polarity_scores(sentence).values()
        return np.array(list(values))

    input_sentences = ['sentence%d'%i for i in range(1,5)]
    first_sentences = df[input_sentences]
    beginning_polarities = first_sentences.apply(lambda x: np.stack(x.apply(polarity).values), axis=1)
    ending = df.drop(columns = input_sentences)
    ending_polarity = ending['sentence5'].apply(polarity)
    X = np.stack(beginning_polarities.values,axis=0)
    Y = np.stack(ending_polarity.values)
    return X,Y

X, y = prepare_roc_dataset(roc_df)
print(y.shape)
trX, teX, trY, teY = train_test_split(X, y, test_size=0.33, random_state=42)

max_features = 1024

model = Sequential()
model.add(LSTM(128))
model.add(Dropout(0.1))
polarity_layer = model.add(Dense(4, activation='softmax'))

def cosine_similarity_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        pred_norm = y_pred/K.batch_dot(y_pred,y_pred, axes=[1,1])
        true_norm = y_true/K.batch_dot(y_true,y_true, axes=[1,1])
        cosine = K.batch_dot(pred_norm, true_norm, axes=[1,1])
        cosine = K.sum(cosine)
        return -cosine
    # Return a function
    return loss

model.compile(loss=cosine_similarity_loss(polarity_layer),
              optimizer='adam')

model.fit(trX, trY, batch_size=16, epochs=10)
score = model.evaluate(teX, teY, batch_size=16)
print(score)

roc_df = pd.read_csv(DATA_DIR + '/' + VAL_SET).drop(columns=['InputStoryid'])
X_test, answer1, answer2, y_test = prepare_test_dataset(roc_df)

predictions = model.predict(X_test, batch_size=1)
correct = 0
for i in range(predictions.shape[0]):
    if cosine_similarity(predictions[i].reshape(1,-1), answer1[i].reshape(1,-1)) \
     < cosine_similarity(predictions[i].reshape(1,-1), answer2[i].reshape(1,-1)):
        y = 2
    else:
        y = 1
    if y == y_test[i]:
        correct+=1

print(correct/ predictions.shape[0])

