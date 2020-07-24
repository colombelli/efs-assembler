import pandas as pd
import pickle
import glob
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import engine.kuncheva_index as ki
from engine.DataManager import DataManager
from engine.Constants import *

class Evaluator:

    # th_in_fraction: bool  => if the threshold values are fractions or integers
    def __init__(self, data_manager:DataManager, thresholds, th_in_fraction):

        self.dm = data_manager
        
        self.thresholds = None
        self.frac_thresholds = None
        self.__init_thresholds(thresholds, th_in_fraction)

        self.current_threshold = None
        self.current_eval_level = None  # 3: second layer aggregated rankings
                                        # 2: first layer aggregated rankings
                                        # 1: first rankings built from single FS methods
        
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
                print("0 int threshold value detected for fraction", th, "- skipping.")
                continue

            updated_fraction_thresholds.append(th)
            int_thresholds.append(int_th)

        print("\nNumber of genes to select given the threshold percentages:")
        print(int_thresholds, "\n\n")
        return int_thresholds, updated_fraction_thresholds


    def get_frac_thresholds(self, thresholds):

        dataset_len = len(self.dm.pd_df.columns)

        updated_int_thresholds = []
        frac_thresholds = []
        for th in thresholds:

            if not(th):
                print("0 int threshold value detected for fraction", th, "- skipping.")
                continue

            if th > dataset_len - 1:
                print("Given threshold value,", str(th)+", is greater the number of features - skipping.")

            updated_int_thresholds.append(th)
            frac_th = (th * 100) / dataset_len
            frac_thresholds.append(frac_th)

        print("\nNumber of genes to select given the threshold percentages:")
        print(updated_int_thresholds, "\n\n")
        return updated_int_thresholds, frac_thresholds


    def __reset_classifier(self):
        self.classifier = SVC(gamma='auto', probability=True)
        return
    
    def __reset_confusion_matrices(self):
        self.confusion_matrices = []
        return


    def __get_gene_lists(self, pd_rankings):
        gene_lists = []

        for ranking in pd_rankings:
            index_names_arr = ranking.index.values
            gene_lists.append(list(index_names_arr))
        
        return gene_lists


    def __get_x(self, df, genes):
        return df.loc[:, genes]
    
    def __get_y(self, df):
        return df.loc[:, ['class']].T.values[0]



    def get_prediction_performance(self):
        
        self.__reset_classifier()
        self.classifier.fit(self.training_x, self.training_y)

        accuracy = self.classifier.score(self.testing_x, self.testing_y)

        pred = self.classifier.predict_proba(self.testing_x)
        pred = self.__get_probs_positive_class(pred)

        roc_auc = metrics.roc_auc_score(np.array(self.testing_y, dtype=int), pred)

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



    def evaluate_final_rankings(self):
        
        self.current_eval_level = 3

        print("Computing stabilities...")
        stabilities = self.__compute_stabilities()

        with open(self.dm.results_path+"fold_sampling.pkl", 'rb') as file:
                folds_sampling = pickle.load(file)
        print("Computing prediction performances...")
        prediction_performances = self.__compute_prediction_performances(folds_sampling)

        self.__save_confusion_matrices(FINAL_CONFUSION_MATRICES_FILE_NAME)
        self.__reset_confusion_matrices()
        self.stabilities = stabilities
        self.prediction_performances = prediction_performances
        
        return self.stabilities, self.prediction_performances
    

    def __compute_stabilities(self, final_rankings_intermediate=None):
             
        th_stabilities = []
        for th in self.thresholds:

            if self.current_eval_level == 3:
                final_rankings = self.__get_final_rankings(th)
            elif self.current_eval_level == 2:
                final_rankings = self.__get_lvl2_rankings(th, final_rankings_intermediate)
            elif self.current_eval_level == 1:
                final_rankings = final_rankings_intermediate


            self.rankings = self.__get_gene_lists(final_rankings)
            th_stabilities.append(self.get_stability(th))

        return th_stabilities

    
    def __get_final_rankings(self, threshold):

        final_rankings = []
        for fold_iteration in range(self.dm.num_folds):
            ranking_path = self.dm.results_path + "fold_" + str(fold_iteration+1) + "/"
            file_path = ranking_path + AGGREGATED_RANKING_FILE_NAME + str(threshold) + ".csv"
            #ranking = self.dm.load_RDS(file)
            #ranking = self.dm.r_to_pandas(ranking)
            ranking = self.dm.load_csv(file_path)
            final_rankings.append(ranking)

        return final_rankings

    
    def __get_lvl2_rankings(self, threshold, lvl2_ranking_paths):
        
        final_rankings = []
        for ranking_path in lvl2_ranking_paths:
            ranking_path += str(threshold) + ".csv"
            ranking = self.dm.load_csv(ranking_path)
            final_rankings.append(ranking)

        return final_rankings


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
                
                final_rankings = self.__get_final_rankings(th)
                self.rankings = self.__get_gene_lists(final_rankings)

                genes = self.rankings[i][0:th]
                self.__set_data_axes(training, testing, genes)
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

    def __set_data_axes(self, training, testing, genes):

        training_df = self.dm.pd_df.loc[training]
        testing_df = self.dm.pd_df.loc[testing]

        self.training_x = self.__get_x(training_df, genes)
        self.training_y = self.__get_y(training_df)
        self.testing_x = self.__get_x(testing_df, genes)
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


    def evaluate_intermediate_hyb_rankings(self):
        
        level1_rankings, level2_rankings = self.__get_intermediate_rankings()
        
        print("\nEvaluating level 1 rankings...")
        self.current_eval_level = 1
        level1_evaluation = self.__evaluate_level1_rankings(level1_rankings)

        print("\n\nEvaluating level 2 rankings...")
        self.current_eval_level = 2
        level2_evaluation = self.__evaluate_intermediate_rankings(level2_rankings)        
        return level1_evaluation, level2_evaluation
        

    def __evaluate_level1_rankings(self, level1_rankings):
        
        level1_evaluation = {}  # a dict where each key is a single fs method
                                # and the value is a intermediate ranking like
                                # evaluation

        for fs_method in level1_rankings:
            print("\nEvaluating", fs_method, "FS method")
            level1_evaluation[fs_method] = self.__evaluate_intermediate_rankings(
                                                        level1_rankings[fs_method])
        return level1_evaluation



    def __evaluate_intermediate_rankings(self, final_rankings):
        
        with open(self.dm.results_path+"fold_sampling.pkl", 'rb') as file:
            folds_sampling = pickle.load(file)

        prediction_performances = {
            ACCURACY_METRIC: [],
            ROC_AUC_METRIC: [],
            PRECISION_RECALL_AUC_METRIC: []
        }
        stabilities = []
        for i, fold_rankings in enumerate(final_rankings):

            print("Computing stabilities...")
            stabilities.append(self.__compute_stabilities(fold_rankings))

            
            print("Computing prediction performances...")
            acc, roc, pr = self.__compute_intermediate_pred_perf(folds_sampling[i], fold_rankings) 
            prediction_performances[ACCURACY_METRIC] += acc
            prediction_performances[ROC_AUC_METRIC] += roc
            prediction_performances[PRECISION_RECALL_AUC_METRIC] += pr
                
        return stabilities, prediction_performances

    
    def __get_intermediate_rankings(self):

        level2_rankings_paths = []  # each item is a list representing each fold iteration
                              # these lists contain the ranking paths for each bootstrap
                              # [
                              # fold1 = [agg_r1, agg_r2, agg_r3, ...],
                              # fold2 = [agg_r1, agg_r2, agg_r3, ...],
                              # fold3 = [agg_r1, agg_r2, agg_r3, ...],
                              #  ...
                              # ] 


        level1_rankings = {}  # each key is a fs method
                              # each value is a level2_rankings kind-of-structure
                              
        # so it looks like:
        # level1_rankings = {
        #              fs1: [
        #                       fold1 = [r1, r2, r3, ...],
        #                       fold2 = [r1, r2, r3, ...],
        #                       fold3 = [r1, r2, r3, ...],
        #                       ...
        #               ], 
        #               fs2: [
        #                       fold1 = [r1, r2, r3, ...],
        #                       fold2 = [r1, r2, r3, ...],
        #                       fold3 = [r1, r2, r3, ...],
        #                       ...
        #               ],
        #               ...         
        # }

        self.__init_level1_rankings_dict(level1_rankings)
        print("Loading level 1 rankings...")
        self.__load_level1_rankings(level1_rankings)
        print("Build level 2 ranking paths...")
        self.__load_level2_ranking_paths(level2_rankings_paths)

        return level1_rankings, level2_rankings_paths
    

    def __init_level1_rankings_dict(self, level1_rankings):

        fs_names = self.__get_single_fs_names()
        for fs_name in fs_names:
            level1_rankings[fs_name] = []
        return


    def __get_single_fs_names(self):
        
        ranking_path = self.__build_ranking_path_string(1, 1)
        single_ranking_file_names = self.__get_single_fs_ranking_file_names(
                                                        ranking_path)
        single_fs_names = []
        for path in single_ranking_file_names:
            single_fs_names.append(
                self.__get_fs_method_name_by_its_path(path)
            )

        return single_fs_names


    def __build_ranking_path_string(self, fold_iteration, bootstrap):
        return self.dm.results_path + "fold_" + str(fold_iteration) + "/" + \
                    "bootstrap_" + str(bootstrap) + "/"


    def __get_fs_method_name_by_its_path(self, path):
        return path.split("/")[-1].split(".")[0]


    def __get_single_fs_ranking_file_names(self, path):
        file_names_style = path + "*.csv"
        return [f for f in glob.glob(f"{file_names_style}") 
                    if AGGREGATED_RANKING_FILE_NAME not in f]
    

    def __load_level1_rankings(self, level1_rankings):
        
        for fs_method in level1_rankings:
            for fold_iteration in range(1, self.dm.num_folds+1):
                
                bs_rankings = []
                for bootstrap in range(1, self.dm.num_bootstraps+1):
                    
                    ranking_path = self.__build_ranking_path_string(fold_iteration, bootstrap)
                    bs_rankings.append(self.__load_single_fs_ranking(ranking_path, fs_method))

                level1_rankings[fs_method].append(bs_rankings)
        return

    
    def __load_level2_ranking_paths(self, level2_ranking_paths):

        for fold_iteration in range(1, self.dm.num_folds+1):
            
            agg_rankings = []
            for bootstrap in range(1, self.dm.num_bootstraps+1):
                
                ranking_path = self.__build_ranking_path_string(fold_iteration, bootstrap)
                ranking_path += AGGREGATED_RANKING_FILE_NAME
                agg_rankings.append(ranking_path)

            level2_ranking_paths.append(agg_rankings)
        return


    def __load_agg_rankings(self, ranking_path):
        file = ranking_path + AGGREGATED_RANKING_FILE_NAME 
        ranking = self.dm.load_RDS(file)
        ranking = self.dm.r_to_pandas(ranking)
        return ranking

    
    def __load_single_fs_ranking(self, ranking_path, fs_method):
        file = ranking_path + fs_method + ".csv"
        ranking = self.dm.load_csv(file)
        return ranking


    
    def __compute_intermediate_pred_perf(self, fold_sampling, fold_rankings):
        
        if self.current_eval_level == 1:
            self.rankings = self.__get_gene_lists(fold_rankings)
        
        training, testing = fold_sampling

        bs_accs = []
        bs_roc_aucs = []
        bs_pr_aucs = []

        th_accs = []
        th_roc_aucs = []
        th_pr_aucs = []
        for th in self.thresholds:

            if self.current_eval_level == 2:
                rankings = self.__get_lvl2_rankings(th, fold_rankings)
                self.rankings = self.__get_gene_lists(rankings)
            
            for ranking in self.rankings:
                genes = ranking[0:th]
                self.__set_data_axes(training, testing, genes)
                acc, roc, pr = self.get_prediction_performance()
                th_accs.append(acc)
                th_roc_aucs.append(roc)
                th_pr_aucs.append(pr)

        bs_accs.append(th_accs)
        bs_roc_aucs.append(th_roc_aucs)
        bs_pr_aucs.append(th_pr_aucs)
        
        return bs_accs, bs_roc_aucs, bs_pr_aucs