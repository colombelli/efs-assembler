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

class ExperimentRecyle:


    """
        experiment object should be a dictionary in the following format:

        {
            "type": <experiment type>,
            "seed": <int>,
            "thresholds": [<int for threshold1>, <int for threshold2>, ... , <int for thresholdn>],
            "bootstraps": <int number of bags for bootstrapping data if it's a Hybrid/Homogeneous ensemble>,
            "aggregators": [<aggregator1 object>, <aggregator2 object>],
            "rankers": [<ranker1 object>, <ranker2 object>, ..., <rankern object>],
            "classifier": "classifier_model_file_name",   --> see ./classifiers/ folder. currently available: "gbc", "svm" and "random_forest"
            "dataset": <path to dataset>,
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

        Note: "thresholds", "aggregators" and "rankers" properties need to be lists, even if
                they have only one element.
        
    """

    def __init__(self, experiment, results_path, base_experiment_path):

        
        self.exp = experiment

        if results_path[-1] != "/":
            self.results_path = results_path + "/"
        else:
            self.results_path = results_path


        if base_experiment_path[-1] != "/":
            self.base_experiment_path = base_experiment_path + "/"
        else:
            self.base_experiment_path = base_experiment_path

        self.num_folds = None
        self.__infer_number_of_folds()



    def __infer_number_of_folds(self):
        path_dirs = None
        for _, subdirs, _ in os.walk(self.base_experiment_path):
            path_dirs = subdirs
            break
        self.num_folds = len([i for i in path_dirs if "fold_" in i])
        return


    def _mount_experiment_folder_name(self, count):
        sufix = "_E" + str(count+1) + "/"
        rad = self.exp["type"] + "_" + self.exp["dataset"].split('/')[-1].split('.')[0]
        return rad+sufix


    def run(self):
        
        exp_count = sum(os.path.isdir(self.results_path+i) for i in os.listdir(self.results_path))

        exp_name = self._mount_experiment_folder_name(exp_count)
        complete_results_path = self.results_path + exp_name

        int_seed = round(int(self.exp["seed"]))

        """
        TO-DO: implement final selection behavior
        
        if ("balanced_final_selection" in exp):
            balanced_final_selection = exp["balanced_final_selection"]
        else:
            Logger.balanced_final_selection_not_specified()
            balanced_final_selection=True
        """

        ths = self.exp["thresholds"]
        classifier_file = self.exp["classifier"]

        if self.exp["type"] == 'sin':
            self.perform_selection_single(self.exp["dataset"], complete_results_path, self.exp["rankers"],
                                            int_seed, ths, classifier_file)

        elif self.exp["type"] == 'hom':
            int_bootstraps = round(int(self.exp["bootstraps"]))
            self.perform_selection_hom(self.exp["dataset"], complete_results_path,
                                        self.exp["rankers"], self.exp["aggregators"][0],
                                        int_bootstraps, int_seed, ths, classifier_file)

        elif self.exp["type"] == 'het':
            self.perform_selection_het(self.exp["dataset"], complete_results_path,
                                        self.exp["rankers"], self.exp["aggregators"][0],
                                        int_seed, ths, classifier_file)
            
        elif self.exp["type"] == 'hyb':
            int_bootstraps = round(int(self.exp["bootstraps"]))
            self.perform_selection_hyb(self.exp["dataset"], complete_results_path, self.exp["rankers"], 
                                        self.exp["aggregators"][0], self.exp["aggregators"][1], 
                                        int_bootstraps, int_seed, ths, classifier_file)
        return


    def compute_time_taken(self, st):
        
        end = time()
        hours, rem = divmod(end-st, 3600)
        minutes, seconds = divmod(rem, 60)
    
        formatted_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        Logger.time_taken(formatted_time_str)
        return


    def perform_selection_hyb(self, dataset_path, results_path, rankers, aggregator1, aggregator2,
                                num_bootstraps, seed, ths, classifier_file):
        
        str_aggregators = [aggregator1, aggregator2]
        str_rankers = [i[0] for i in rankers]

        dm = DataManager(results_path, dataset_path, num_bootstraps, self.num_folds, 
                        undersampling=True, seed=seed, base_experiment_path=self.base_experiment_path)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        dm.init_data_folding_process()

        ev = Evaluator(dm, ths, classifier_file)
        im = InformationManager(dm, ev, str_rankers, str_aggregators)
        ensemble = Hybrid(dm, rankers, aggregator1, aggregator2, ths, experiment_recycle=True)

        st = time()
        ensemble.select_features_experiment()
        self.compute_time_taken(st)

        Logger.decoding_dataframe()
        dm.decode_main_dm_df()

        Logger.starting_evaluation_process()
        ev.evaluate_final_ranks()

        Logger.creating_csv_files()
        im.create_csv_tables()
        
        # TO-DO:
        #final_selection = FinalSelection(ensemble, balanced_final_selection)
        #final_selection.start()
        # also have to update: dm.update_bootstraps_outside_cross_validation()

        Logger.end_experiment_message()
        return


    def perform_selection_het(self, dataset_path, results_path, rankers, aggregator, 
                                    seed, ths, classifier_file):
        return

    

    def perform_selection_hom(self, dataset_path, results_path, ranker, aggregator, 
                                    num_bootstraps, seed, ths, classifier_file):

        return

    

    def perform_selection_single(self, dataset_path, results_path, ranker, seed, 
                                        ths, classifier_file):
        return