#!/bin/env python

# An idea for a common API followed by all models.
# All the models are supposed to extend and implement the abstract class
# Model with the following API (according to the sklear model API)
#
# model = SentimentLSTM()
# X, y = model.get_training_data()
# model.fit(X, y)
# X_test, y_true = model.get_testing_data()
# y_pred = model.predict(X_test)
# model.evaluate(y_true, y_pred)
# predictions[model.name] = y_pred
# np.save('sentiment_predictions.npy', y_pred)
#
# The final script will contain the above code as many times as the models
# and maybe a final call to the ensemble model which uses as features the
# predictions of all the the other models
#

from sklearn.metrics import accuracy_score

class NLUModel():

    def __init__(self, name):
        self.name = name
        self.__build()

    def __build(self):
        """
        Build the graph of the model.
        """
        pass

    def fit(self, X, y, epochs=10, batch_size=32):
        pass

    def predict(self, X):
        """
        Returns: a np.array of shape(validation_set.shape[0],) with the
        predictions of the model.
        NOTICE: The predictions must be 0 or 1!
        """
        pass

    def get_train_data(self):
        """
        Returns: (X_train, y_train) as they are supposed to be fed to the fit function.
        """
        pass

    def get_test_data(self):
        """
        Returns: (X_test, y_test) as they are supposed to be fed to the predict function and
        the evaluate function.
        """
        pass

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Returns: a float corresponding to the accuracy of the model.
        """
        score = accuracy_score(y_true, y_pred)

        return score


