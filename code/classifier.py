import numpy as np
from sklearn import metrics

class MLTrainer():
    def __init__(self, x_train, y_train, x_test, y_test, training_algorithm, normalise = False, **kwargs):
        self.normalise = normalise
        if self.normalise == True:
            y_train = self.normalise_data(y_train)
            y_test = self.normalise_data(y_test)
        else:
            y_train = self.one_hot_encoding(y_train)
            y_test = self.one_hot_encoding(y_test)
            
        self.y_train = y_train
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.training_algorithm = training_algorithm

        self.classifier_train()
        self.classifier_pred()
        self.classifier_results()

    def classifier_train(self):
        clf = self.training_algorithm()
        clf.fit(self.x_train, self.y_train) 
        self.clf = clf

    def classifier_pred(self):
        y_pred = self.clf.predict(self.x_test)

        if self.normalise == True:
            y_pred = self.normalise_data(y_pred)
        else:
            y_pred = self.one_hot_encoding(y_pred)
        self.y_pred = y_pred

    def classifier_results(self):

        results = {}

        results["f1"] = round(metrics.f1_score(self.y_test, self.y_pred, average='macro'),4)
        results["accuracy"] = round(metrics.accuracy_score(self.y_test, self.y_pred), 4)
        self.results = results
    
    def get_classifier_pred(self):
        return self.y_pred
    
    def get_classifier_results(self):
        return self.results
    
    @staticmethod
    def normalise_data(y):
        if np.max(y) >= 1:
            return y/np.max(y)
        return y

    @staticmethod
    def one_hot_encoding(y):
        return y - np.min(y)
