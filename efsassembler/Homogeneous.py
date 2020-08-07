from efsassembler.Logger import Logger
from efsassembler.Selector import FSelector, PySelector, RSelector
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME, SELECTION_PATH

class Homogeneous:
    
    # fs_method: a single elemet list (to maintain coherence) whose element is a tuple: 
    # (script name, language which the script was written, .csv output name)
    def __init__(self, data_manager:DataManager, fs_method, aggregator, thresholds:list):

        self.dm = data_manager
        self.fs_method = FSelector.generate_fselectors_object(fs_method)[0]
        self.aggregator = Aggregator(aggregator)
        self.rankings_to_aggregate = None

        self.thresholds = thresholds
        self.current_threshold = None



    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()

            rankings = []
            for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
                Logger.bootstrap_fold_iteration(j+1, i+1)
                
                output_path = self.dm.get_output_path(i, j)
                bootstrap_data = self.dm.pd_df.iloc[bootstrap]
                rankings.append(self.fs_method.select(bootstrap_data, output_path))

                
            self.__set_rankings_to_aggregate(rankings)

            output_path = self.dm.get_output_path(fold_iteration=i)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
            for th in self.thresholds:
                Logger.aggregating_rankings()
                Logger.for_threshold(th)
                self.current_threshold = th
                aggregation = self.aggregator.aggregate(self)
                
                self.dm.save_encoded_ranking(aggregation, file_path+str(th)) 
        return


    def __set_rankings_to_aggregate(self, rankings):
        self.rankings_to_aggregate = rankings
        return


    def select_features(self):

        Logger.whole_dataset_selection()
        self.dm.update_bootstraps_outside_cross_validation()

        rankings = []
        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            Logger.bootstrap_iteration(j+1)
            
            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
            rankings.append(self.fs_method.select(bootstrap_data, save_ranking=False))

        self.__set_rankings_to_aggregate(rankings)

        output_path = self.dm.results_path + SELECTION_PATH
        file_path = output_path + AGGREGATED_RANKING_FILE_NAME
        for th in self.thresholds:
            Logger.aggregating_rankings()
            Logger.for_threshold(th)
            self.current_threshold = th
            aggregation = self.aggregator.aggregate(self)
            
            self.dm.save_encoded_ranking(aggregation, file_path+str(th)) 
        return
