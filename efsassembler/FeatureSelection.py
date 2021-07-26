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

class FeatureSelection:

    """
        selection_cfgs object should be a list of dictionaries in the following format:

        {
            "type": <selection type>,
            "seed": <int>,
            "thresholds": [<int for threshold1>, <int for threshold2>, ... , <int for thresholdn>],
            "bootstraps": <int number of bags for bootstrapping data if it's a Hybrid/Homogeneous ensemble>,
            "aggregators": [<aggregator1 object>, <aggregator2 object>],
            "rankers": [<ranker1 object>, <ranker2 object>, ..., <rankern object>],
            "datasets": [<path to dataset1>, <path to dataset2>, ..., <path to dataset n>],
            "balanced_selection": <bool indicating if feature selection is to be applied in balanced folds> 
        }

        <selection type>: 
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


        If 'balanced_selection' is not given, True is assumed.

        Note: "thresholds", "aggregators", "rankers" and "datasets" properties need to be lists, even if
                they have only one element. The same goes for selection_cfgs object itself. 
    """

    def __init__(self, selection_cfgs, results_path):

        
        self.selection_cfgs = selection_cfgs
        self._should_load_FSelectorRcpp()

        if results_path[-1] != "/":
            self.results_path = results_path + "/"
        else:
            self.results_path = results_path


    def _mount_selection_folder_name(self, i, count, cfg, ds_path):

        sufix = "_Selection" + str(count+i+1) + "/"
        rad = cfg["type"] + "_" + ds_path.split('/')[-1].split('.')[0]
        return rad+sufix


    def _should_load_FSelectorRcpp(self):
        for configuration in self.selection_cfgs:
            for ranker in configuration["rankers"]:
                if ranker[1] == 'r':
                    rpackages.quiet_require('FSelectorRcpp')
                    return
        return



    def run(self):
        
        cfg_count = sum(os.path.isdir(self.results_path+i) for i in os.listdir(self.results_path))


        for i, cfg in enumerate(self.selection_cfgs):
            for dataset_path in cfg["datasets"]:
                cfg_name = self._mount_selection_folder_name(i, cfg_count, cfg, dataset_path)
                complete_results_path = self.results_path + cfg_name

                int_seed = round(int(cfg["seed"]))
  
                if ("balanced_selection" in cfg):
                    balanced_selection = cfg["balanced_selection"]
                else:
                    Logger.balanced_selection_not_specified()
                    balanced_selection=True

                ths = cfg["thresholds"]

                if cfg["type"] == 'sin':
                    dm = self.create_data_manager_object(dataset_path, complete_results_path, 0, int_seed)
                    fs_technique = SingleFR(dm, cfg["rankers"], ths)
                    str_aggregators = None


                elif cfg["type"] == 'hom':
                    int_bootstraps = round(int(cfg["bootstraps"]))
                    dm = self.create_data_manager_object(dataset_path, complete_results_path, int_bootstraps, int_seed)
                    fs_technique = Homogeneous(dm, cfg["rankers"], cfg["aggregators"][0], ths)
                    str_aggregators = [cfg["aggregators"][0]]


                elif cfg["type"] == 'het':
                    dm = self.create_data_manager_object(dataset_path, complete_results_path, 0, int_seed)
                    fs_technique = Heterogeneous(dm, cfg["rankers"], cfg["aggregators"][0], ths)
                    str_aggregators = [cfg["aggregators"][0]]
                    
                    
                elif cfg["type"] == 'hyb':
                    int_bootstraps = round(int(cfg["bootstraps"]))
                    dm = self.create_data_manager_object(dataset_path, complete_results_path, int_bootstraps, int_seed)
                    fs_technique = Hybrid(dm, cfg["rankers"], cfg["aggregators"][0], cfg["aggregators"][1], ths)
                    str_aggregators = [cfg["aggregators"][0], cfg["aggregators"][1]]
                    

                str_rankers = [i[0] for i in cfg["rankers"]]
                InformationManager(dm, None, str_rankers, str_aggregators)  #creates text file info
                self.perform_selection(fs_technique, balanced_selection)
                Logger.end_feature_selection_message()
        return


    def compute_time_taken(self, st):
        
        end = time()
        hours, rem = divmod(end-st, 3600)
        minutes, seconds = divmod(rem, 60)
    
        formatted_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        Logger.time_taken(formatted_time_str)
        return


    def create_data_manager_object(self, dataset_path, results_path, num_bootstraps, seed):
        dm = DataManager(results_path, dataset_path, num_bootstraps, 0, seed=seed)
        Logger.encoding_dataset()
        dm.encode_main_dm_df()
        dm.create_results_dir()
        return dm


    def perform_selection(self, fs_technique, balanced_selection):
        st = time()
        final_selection = FinalSelection(fs_technique, balanced_selection)
        final_selection.start(skip_encoding=True)
        self.compute_time_taken(st)
        return