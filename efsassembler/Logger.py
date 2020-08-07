import sys
import logging



class Logger:

    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
    handler = logging.getLogger("efs-assembler")

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