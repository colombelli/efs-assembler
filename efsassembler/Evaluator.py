import pandas as pd
import pickle
import glob
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import metrics
import numpy as np
import efsassembler.kuncheva_index as ki
from efsassembler.Logger import Logger
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANK_FILE_NAME, SINGLE_RANK_FILE_NAME, FINAL_CONFUSION_MATRICES_FILE_NAME, ACCURACY_METRIC, ROC_AUC_METRIC, PRECISION_RECALL_AUC_METRIC

class Evaluator:

    # th_in_fraction: bool  => if the threshold values are fractions or integers
    def __init__(self, data_manager:DataManager, thresholds, th_in_fraction):

        self.dm = data_manager
        
        self.thresholds = None
        self.frac_thresholds = None
        self.__init_thresholds(thresholds, th_in_fraction)
        
        self.final_ranks = None     # If the ranks are threshold sensitive, than they will be loaded
                                    # at each iteration to save memory
        self.is_agg_th_sensible = None

        self.current_threshold = None
        self.classifier = None
        self.prediction_performances = None
        self.stabilities = None
        self.rankings = None
        self.training_x = None
        self.training_y = None
        self.testing_x = None
        self.testing_y = None

        self.confusion_matrices = []
        

    
    def __init_thresholds(self, thresholds, th_in_fraction):
        if th_in_fraction:
            self.thresholds, self.frac_thresholds = self.get_int_thresholds(thresholds)
        else:
            self.thresholds, self.frac_thresholds = self.get_frac_thresholds(thresholds)
        return


    def get_int_thresholds(self, thresholds):

        dataset_len = len(self.dm.pd_df.columns)

        updated_fraction_thresholds = []
        int_thresholds = []
        for th in thresholds:

            int_th = int(dataset_len * th/100)
            if not(int_th):
                Logger.zero_int_threshold(th)
                continue

            updated_fraction_thresholds.append(th)
            int_thresholds.append(int_th)

        Logger.integer_number_of_thresholds(int_thresholds)
        return int_thresholds, updated_fraction_thresholds


    def get_frac_thresholds(self, thresholds):

        dataset_len = len(self.dm.pd_df.columns)

        updated_int_thresholds = []
        frac_thresholds = []
        for th in thresholds:

            if not(th):
                Logger.zero_int_threshold(th)
                continue

            if th > dataset_len - 1:
                Logger.int_threshold_greater_than_dataset(th)

            updated_int_thresholds.append(th)
            frac_th = (th * 100) / dataset_len
            frac_thresholds.append(frac_th)

        Logger.integer_number_of_thresholds(updated_int_thresholds)
        return updated_int_thresholds, frac_thresholds


    def __infer_if_agg_th_sensible(self):

        file_name = Path(self.dm.results_path + "fold_1/" + \
                        AGGREGATED_RANK_FILE_NAME + str(self.thresholds[0]) + ".csv")
        if file_name.is_file():
            self.is_agg_th_sensible = True
        else:
            self.is_agg_th_sensible = False
        return


    def __load_final_ranks(self):
        final_ranks = []
        for fold_iteration in range(self.dm.num_folds):
            ranking_path = self.dm.results_path + "fold_" + str(fold_iteration+1) + "/"
            file_path = ranking_path + SINGLE_RANK_FILE_NAME + ".csv"
            ranking = self.dm.load_csv(file_path)
            final_ranks.append(ranking)
            
        self.final_ranks = final_ranks
        return


    def __reset_classifier(self):
        #self.classifier = SVC(gamma='auto', probability=True)
        self.classifier = GBC()
        return
    
    def __reset_confusion_matrices(self):
        self.confusion_matrices = []
        return


    def __get_feature_lists(self, pd_rankings):
        feature_lists = []
        for ranking in pd_rankings:
            index_names_arr = ranking.index.values
            feature_lists.append(list(index_names_arr))
        return feature_lists


    def __get_x(self, df, features):
        return df.loc[:, features]
    
    def __get_y(self, df):
        return df.loc[:, ['class']].T.values[0]



    def get_prediction_performance(self):
        
        self.__reset_classifier()
        self.classifier.fit(self.training_x, self.training_y)

        #accuracy = self.classifier.score(self.testing_x, self.testing_y)

        pred = self.classifier.predict_proba(self.testing_x)
        y_pred = np.argmax(pred, axis=1)
        pred = self.__get_probs_positive_class(pred)

        roc_auc = metrics.roc_auc_score(np.array(self.testing_y, dtype=int), pred)
        accuracy = metrics.accuracy_score(self.testing_y, y_pred)

        precision, recall, _ = metrics.precision_recall_curve(np.array(self.testing_y, dtype=int), pred)
        pr_auc = metrics.auc(recall, precision)
        
        return accuracy, roc_auc, pr_auc


    def __get_probs_positive_class(self, pred):
        positive_probs = []

        for prediction in pred:
            positive_probs.append(prediction[1])
        return positive_probs


    def get_stability(self, threshold):
        return ki.get_kuncheva_index(self.rankings, threshold=threshold)



    def evaluate_final_ranks(self):
        self.dm.set_seed()  # reset seed internal state allowing reproducibility for the
                            # evaluation process alone without performing feature selection 
        
        self.__infer_if_agg_th_sensible()
        if not self.is_agg_th_sensible:
            self.__load_final_ranks()

        Logger.computing_stabilities()
        stabilities = self.__compute_stabilities()

        with open(self.dm.results_path+"fold_sampling.pkl", 'rb') as file:
                folds_sampling = pickle.load(file)
        Logger.computing_prediction_performances()
        prediction_performances = self.__compute_prediction_performances(folds_sampling)

        self.__save_confusion_matrices(FINAL_CONFUSION_MATRICES_FILE_NAME)
        self.__reset_confusion_matrices()
        self.stabilities = stabilities
        self.prediction_performances = prediction_performances
        
        return self.stabilities, self.prediction_performances
    

    def __compute_stabilities(self, final_ranks_intermediate=None):

        th_stabilities = []
        for th in self.thresholds:
            final_ranks = self.__get_final_ranks(th)
            self.rankings = self.__get_feature_lists(final_ranks)
            th_stabilities.append(self.get_stability(th))

        return th_stabilities

    
    def __get_final_ranks(self, threshold):

        if not self.is_agg_th_sensible:
            return self.final_ranks

        final_ranks = []
        for fold_iteration in range(self.dm.num_folds):
            ranking_path = self.dm.results_path + "fold_" + str(fold_iteration+1) + "/"
            file_path = ranking_path + AGGREGATED_RANK_FILE_NAME + str(threshold) + ".csv"
            ranking = self.dm.load_csv(file_path)
            final_ranks.append(ranking)

        return final_ranks


    def __compute_prediction_performances(self, folds_sampling):
        
        prediction_performances = {
            ACCURACY_METRIC: [],
            ROC_AUC_METRIC: [],
            PRECISION_RECALL_AUC_METRIC: []
        }

        for i, (training, testing) in enumerate(folds_sampling):

            th_accuracies = []
            th_roc_aucs = []
            th_pr_aucs = []
            th_conf_matrices = {}

            for th in self.thresholds:
                
                final_ranks = self.__get_final_ranks(th)
                self.rankings = self.__get_feature_lists(final_ranks)

                features = self.rankings[i][0:th]
                self.__set_data_axes(training, testing, features)
                acc, roc, pr = self.get_prediction_performance()
                th_accuracies.append(acc)
                th_roc_aucs.append(roc)
                th_pr_aucs.append(pr)
                th_conf_matrices[th] = self.__compute_confusion_matrix()

            prediction_performances[ACCURACY_METRIC].append(th_accuracies)
            prediction_performances[ROC_AUC_METRIC].append(th_roc_aucs)
            prediction_performances[PRECISION_RECALL_AUC_METRIC].append(th_pr_aucs)
            self.confusion_matrices.append(th_conf_matrices)
        
        return prediction_performances

    def __set_data_axes(self, training, testing, features):

        training_df = self.dm.pd_df.iloc[training]
        testing_df = self.dm.pd_df.iloc[testing]

        self.training_x = self.__get_x(training_df, features)
        self.training_y = self.__get_y(training_df)
        self.testing_x = self.__get_x(testing_df, features)
        self.testing_y = self.__get_y(testing_df)
        return


    def __compute_confusion_matrix(self):
        predicted = self.classifier.predict(self.testing_x)
        real = self.testing_y
        confusion_matrix = metrics.confusion_matrix(real, predicted)
        return confusion_matrix

    def __save_confusion_matrices(self, file_name):
        path = self.dm.results_path + file_name
        with open(path, 'wb') as handle:
            pickle.dump(self.confusion_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return