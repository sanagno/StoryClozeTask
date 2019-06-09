#!/bin/env python

from sentimentLSTM import SentimentLSTM
from contextLSTM import ContextLSTM
import numpy as np
import keras

# A dictionary of all the predictions
predictions = {}

def main():

    # ******** Sentiment Analysis *********
    sent_model = SentimentLSTM()
    trX, trY = sent_model.get_train_data()
    teX, teY = sent_model.get_test_data()
    sent_model.fit(trX, trY, epochs=10)
    predictions[sent_model.name] = sent_model.predict(teX)
    np.savetxt('../results/sentiment_lstm.out', predictions[sent_model.name])
    keras.backend.clear_session()

    # ********* Predict Context LSTM *************
    cont_model = ContextLSTM()
    trX, trY = cont_model.get_train_data()
    teX, teY = cont_model.get_test_data()
    cont_model.fit(trX, trY, epochs=10)
    predictions[cont_model.name] = cont_model.predict(teX)
    np.savetxt('../results/context_lstm.out', predictions[cont_model.name])

if __name__ == "__main__":
    main()
