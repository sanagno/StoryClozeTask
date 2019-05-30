#!/bin/env python

from sentimentLSTM import SentimentLSTM
from contextLSTM import ContextLSTM

# A dictionary of all the predictions
predictions = {}

def main():

    # ******** Sentiment Analysis *********
    sent_model = SentimentLSTM()
    trX, trY = sent_model.get_train_data()
    teX, teY = sent_model.get_test_data()
    sent_model.fit(trX, trY, epochs=1)
    predictions[sent_model.name] = sent_model.predict(teX)

    # ********* Predict Context LSTM *************
    cont_model = ContextLSTM()
    trX, trY = cont_model.get_train_data()
    teX, teY = cont_model.get_test_data()
    cont_model.fit(trX, trY, epochs=1)
    predictions[cont_model.name] = cont_model.predict(teX)

if __name__ == "__main__":
    main()