import sys
import os
from time import time

from efsassembler.Logger import Logger
from efsassembler.DataManager import DataManager
from efsassembler.Hybrid import Hybrid
from efsassembler.Heterogeneous import Heterogeneous
from efsassembler.Homogeneous import Homogeneous
from efsassembler.SingleFR import SingleFR
from efsassembler.Evaluator import Evaluator
from efsassembler.InformationManager import InformationManager
from efsassembler.FinalSelection import FinalSelection

import rpy2.robjects.packages as rpackages

class Experiments:

    """
        experiments object should be a list of dictionaries in the following format:

        {
            "type": <experiment type>,
            "seed": <int>,
            "thresholds": [<int for threshold1>, <int for threshold2>, ... , <int for thresholdn>],
            "folds": <int number of folds for the StratifiedKFold cross-validation>,
            "undersampling": <bool indicating if undersampling is to be applied in the stratified cv>,
            "bootstraps": <int number of bags for bootstrapping data if it's a Hybrid/Homogeneous ensemble>,
            "aggregators": [<aggregator1 object>, <aggregator2 object>],
            "rankers": [<ranker1 object>, <ranker2 object>, ..., <rankern object>],
            "classifier": "classifier_model_file_name",   --> see ./classifiers/ folder. currently available: "gbc", "svm" and "random_forest"
            "datasets": [<path to dataset1>, <path to dataset2>, ..., <path to dataset n>],
            "balanced_final_selection": <bool indicating if final feature selection is to be applied in balanced folds> 
        }

        <experiment type>: 
            "sin" for single ranker (no ensemble technique applied, just the chosen feature ranker)
            "het" for heterogeneous ensemble
            "hom" for homogeneous ensemble
            "hyb" for hybrid ensemble

        <agreggator object>:
            "aggregation_method_file_name"

        <ranker object>:
            a tuple: ("ranker_method_file_name", <programming language object>, "rank_file_name.rds")

        <programming language object>:
            either "python" or "r"


        If 'undersampling' and 'balanced_final_selection' are not given, True is assumed.

        Note: "thresholds", "aggregators", "rankers" and "datasets" properties need to be lists, even if
                they have only one element. The same goes for experiments object itself. 
    """

    def __init__(self, experiments, results_path):

        
        self.experiments = experiments
        self._should_load_FSelectorRcpp()

        if results_path[-1] != "/":
            self.results_path = results_path + "/"
        else:
            self.results_path = results_path


    def _mount_experiment_folder_name(self, i, count, exp, ds_path):

        sufix = "_E" + str(count+i+1) + "/"
        rad = exp["type"] + "_" + ds_path.split('/')[-1].split('.')[0]
        return rad+sufix


    def _should_load_FSelectorRcpp(self):
        for experiment in self.experiments:
            for ranker in experiment["rankers"]:
                if ranker[1] == 'r':
                    rpackages.quiet_require('FSelectorRcpp')
                    return
        return



    def run(self):
        
        exp_count = sum(os.path.isdir(self.results_path+i) for i in os.listdir(self.results_path))


        for i, exp in enumerate(self.experiments):
            for dataset_path in exp["datasets"]:
                exp_name = self._mount_experiment_folder_name(i, exp_count, exp, dataset_path)
                complete_results_path = self.results_path + exp_name

                int_folds = round(int(exp["folds"]))
                int_seed = round(int(exp["seed"]))

                if ("undersampling" in exp): 
                    undersampling = exp["undersampling"]
                else: 
                    Logger.undersampling_not_specified()
                    undersampling=True
                if ("balanced_final_selection" in exp):
                    balanced_final_selection = exp["balanced_final_selection"]
                else:
                    Logger.balanced_final_selection_not_specified()
                    balanced_final_selection=True


                ths = exp["thresholds"]
                classifier_file = exp["classifier"]

                if exp["type"] == 'sin':
                    self.perform_selection_single(dataset_path, complete_results_path, exp["rankers"],
                                                    int_folds, undersampling, int_seed, ths, classifier_file,
                                                    balanced_final_selection)

                elif exp["type"] == 'hom':
                    int_bootstraps = round(int(exp["bootstraps"]))
                    self.perform_selection_hom(dataset_path, complete_results_path,
                                                exp["rankers"], exp["aggregators"][0], int_folds, undersampling,
                                                int_bootstraps, int_seed, ths, classifier_file, balanced_final_selection)

                elif exp["type"] == 'het':
                    self.perform_selection_het(dataset_path, complete_results_path,
                                                exp["rankers"], exp["aggregators"][0], int_folds, undersampling,
                                                int_seed, ths, classifier_file, balanced_final_selection)
                    
                elif exp["type"] == 'hyb':
                    int_bootstraps = round(int(exp["bootstraps"]))
                    self.perform_selection_hyb(dataset_path, complete_results_path, exp["rankers"], 
                                                exp["aggregators"][0], exp["aggregators"][1], 
                                                int_folds, undersampling, int_bootstraps, int_seed, ths, classifier_file,
                                                balanced_final_selection)
        return


    def compute_time_taken(self, st):
        
        end = time()
        hours, rem = divmod(end-st, 3600)
        minutes, seconds = divmod(rem, 60)
    
        formatted_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        Logger.time_taken(formatted_time_str)
        
        return



    def perform_selection_hyb(self, dataset_path, results_path, rankers, aggregator1, aggregator2,
                                num_folds, undersampling, num_bootstraps, seed, ths, classifier_file,
                                balanced_final_selection):
        
        str_aggregators = [aggregator1, aggregator2]
        str_rankers = [i[0] for i in rankers]

        dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, undersampling=undersampling, seed=seed)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        dm.init_data_folding_process()

        ev = Evaluator(dm, ths, classifier_file)
        im = InformationManager(dm, ev, str_rankers, str_aggregators)
        ensemble = Hybrid(dm, rankers, aggregator1, aggregator2, ths)

        st = time()
        ensemble.select_features_experiment()
        self.compute_time_taken(st)

        Logger.decoding_dataframe()
        dm.decode_main_dm_df()

        Logger.starting_evaluation_process()
        ev.evaluate_final_ranks()

        Logger.creating_csv_files()
        im.create_csv_tables()
        
        final_selection = FinalSelection(ensemble, balanced_final_selection)
        final_selection.start()

        Logger.end_experiment_message()
        return


    def perform_selection_het(self, dataset_path, results_path, rankers, aggregator, num_folds, undersampling, 
                                    seed, ths, classifier_file, balanced_final_selection):

        str_aggregators = [aggregator]
        str_rankers = [i[0] for i in rankers]
        num_bootstraps = 0

        dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, undersampling=undersampling, seed=seed)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        dm.init_data_folding_process()
        
        ev = Evaluator(dm, ths, classifier_file)
        im = InformationManager(dm, ev, str_rankers, str_aggregators)
        ensemble = Heterogeneous(dm, rankers, aggregator, ths)

        st = time()
        ensemble.select_features_experiment()
        self.compute_time_taken(st)

        Logger.decoding_dataframe()
        dm.decode_main_dm_df()

        Logger.starting_evaluation_process()
        ev.evaluate_final_ranks()

        Logger.creating_csv_files()
        im.create_csv_tables()

        final_selection = FinalSelection(ensemble, balanced_final_selection)
        final_selection.start()

        Logger.end_experiment_message()
        return

    

    def perform_selection_hom(self, dataset_path, results_path, ranker, aggregator, num_folds, undersampling, 
                                    num_bootstraps, seed, ths, classifier_file, balanced_final_selection):

        str_aggregators = [aggregator]
        str_rankers = [ranker[0][0]]

        dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, undersampling=undersampling, seed=seed)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        dm.init_data_folding_process()

        ev = Evaluator(dm, ths, classifier_file)
        im = InformationManager(dm, ev, str_rankers, str_aggregators)
        ensemble = Homogeneous(dm, ranker, aggregator, ths)

        st = time()
        ensemble.select_features_experiment() 
        self.compute_time_taken(st)

        Logger.decoding_dataframe()
        dm.decode_main_dm_df()

        Logger.starting_evaluation_process()
        ev.evaluate_final_ranks()

        Logger.creating_csv_files()
        im.create_csv_tables()

        final_selection = FinalSelection(ensemble, balanced_final_selection)
        final_selection.start()

        Logger.end_experiment_message()
        return

    

    def perform_selection_single(self, dataset_path, results_path, ranker, num_folds, undersampling, seed, 
                                        ths, classifier_file, balanced_final_selection):

        num_bootstraps = 0
        str_rankers = [ranker[0][0]]    # because ranker is always a list, even when it have only one element

        dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, undersampling=undersampling, seed=seed)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        dm.init_data_folding_process()

        ev = Evaluator(dm, ths, classifier_file)
        im = InformationManager(dm, ev, str_rankers)
        feature_ranker = SingleFR(dm, ranker, ths)

        st = time()
        feature_ranker.select_features_experiment()
        self.compute_time_taken(st)

        Logger.decoding_dataframe()
        dm.decode_main_dm_df()

        Logger.starting_evaluation_process()
        ev.evaluate_final_ranks()

        Logger.creating_csv_files()
        im.create_csv_tables()

        final_selection = FinalSelection(feature_ranker, balanced_final_selection)
        final_selection.start()

        Logger.end_experiment_message()
        return

