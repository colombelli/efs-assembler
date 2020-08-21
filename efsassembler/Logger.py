import sys
import logging
import os


class Logger:

    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
    handler = logging.getLogger("efs-assembler")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disables tf logs
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    @classmethod
    def time_taken(self, formatted_time_string):
        log = "Feature selection step time taken: " + formatted_time_string
        Logger.handler.info(log)
        return
    
    @classmethod
    def end_experiment_message(self):
        Logger.handler.info("Experiment finished!")
        return

    @classmethod
    def decoding_dataframe(self):
        Logger.handler.info("Decoding dataframe")
        return

    @classmethod
    def starting_evaluation_process(self):
        Logger.handler.info("Starting evaluation process")
        return

    @classmethod
    def creating_csv_files(self):
        Logger.handler.info("Creating csv files")
        return

    @classmethod
    def evaluating_inner_levels(self):
        Logger.handler.info("Evaluating inner levels")
        return

    @classmethod
    def ranking_features_with_script(self, script_string):
        Logger.handler.info("Ranking features with " + script_string)
        return

    @classmethod
    def fold_iteration(self, fold_iteration):
        Logger.handler.info("Starting fold iteration " + str(fold_iteration))
        return
    
    @classmethod
    def whole_dataset_selection(self):
        Logger.handler.info("Selecting features using the whole dataset")
        return
    
    @classmethod
    def for_threshold(self, threshold):
        Logger.handler.info("Threshold: " + str(threshold))
        return

    @classmethod
    def aggregating_rankings(self):
        Logger.handler.info("Aggregating rankings")
        return
    
    @classmethod
    def aggregating_n_level_rankings(self, level):
        log = "Aggregating level " + str(level) + " rankings"
        Logger.handler.info(log)
        return

    @classmethod
    def bootstrap_fold_iteration(self, bootstrap_it, fold_it):
        log = "Bootstrap: " + str(bootstrap_it) + " | Fold iteration: " + str(fold_it)
        Logger.handler.info(log)
        return

    @classmethod
    def bootstrap_iteration(self, bootstrap_it):
        Logger.handler.info("Bootstrap: " + str(bootstrap_it))
        return

    @classmethod    #Maybe change this to Warning instead
    def zero_int_threshold(self, fraction_threshold):
        log = "0 int threshold value detected for fraction " + str(fraction_threshold) + " - skipping."
        Logger.handler.info(log)
        return

    @classmethod    #Maybe change this to Warning instead
    def int_threshold_greater_than_dataset(self, threshold):
        log = "Given threshold value, " + str(threshold) + ", is greater the number of available features - skipping."
        Logger.handler.info(log)
        return

    @classmethod
    def integer_number_of_thresholds(self, integer_thresholds):
        log = "Number of features to select given the threshold percentages: " + str(integer_thresholds)
        Logger.handler.info(log)
        return

    @classmethod
    def computing_stabilities(self):
        Logger.handler.info("Computing stabilities")
        return
    
    @classmethod
    def computing_prediction_performances(self):
        Logger.handler.info("Computing prediction performances")
        return

    @classmethod
    def evaluating_n_level(self, n):
        log = "Evaluating level " + str(n) + " rankings"
        Logger.handler.info(log)
        return

    @classmethod
    def evaluating_x_fs_method(self, fs_method):
        log = "Evaluating " + str(fs_method) + " FS method"
        Logger.handler.info(log)
        return

    @classmethod
    def loading_lvl1_rankings(self):
        Logger.handler.info("Loading level 1 rankings")
        return

    @classmethod
    def loading_lvl2_ranking_paths(self):
        Logger.handler.info("Loading level 2 ranking paths")
        return

    @classmethod
    def loading_x_dataset(self, dataset_path):
        log = "Loading dataset: " + dataset_path
        Logger.handler.info(log)
        return

    @classmethod    
    def creating_results_directory_in_x_path(self, path):
        log = "Creating results directory in: " + path
        Logger.handler.info(log)
        return

    @classmethod
    def create_inner_results_csv_files(self):
        Logger.handler.info("Creating intermediate results csv files")
        return

    @classmethod
    def encoding_dataset(self):
        Logger.handler.info("Encoding dataset")
        return

    @classmethod
    def decoding_dataset(self):
        Logger.handler.info("Decoding dataset")
        return