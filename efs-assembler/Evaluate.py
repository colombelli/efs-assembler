import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np
import engine.kuncheva_index as ki

class Evaluate:

    def __init__(self, rankings, threshold, trainingDF, testingDF):

        self.rankings = self.__get_gene_lists(rankings)
        self.threshold = threshold
        self.training_x = self.__get_x(trainingDF)
        self.training_y = self.__get_y(trainingDF)
        self.testing_x = self.__get_x(testingDF)
        self.testing_y = self.__get_y(testingDF)


    def __get_gene_lists(self, pdRankings):
        gene_lists = []

        for ranking in pdRankings:
            index_names_arr = ranking.index.values
            gene_lists.append(list(index_names_arr))
        
        return gene_lists


    def __get_x(self, df):
        return df.loc[:, df.columns != 'class']
    
    def __get_y(self, df):
        return df.loc[:, ['class']].T.values[0]



    def get_auc(self):
        
        clf = SVC(gamma='auto', probability=True)
        clf.fit(self.training_x, self.training_y)

        y = self.testing_y
        pred = clf.predict_proba(self.testing_x)
        pred = self.__get_probs_positive_class(pred)

        return metrics.roc_auc_score(np.array(y, dtype=int), pred)


    def __get_probs_positive_class(self, pred):
        positive_probs = []

        for prediction in pred:
            positive_probs.append(prediction[1])
        return positive_probs


    def get_stability(self):
        return ki.get_kuncheva_index(self.rankings, threshold=self.threshold)

