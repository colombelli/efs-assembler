from efsassembler.Logger import Logger
from efsassembler.Selector import FSelector, PySelector, RSelector
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME, SELECTION_PATH

class Heterogeneous:
    
    # fs_methods: a tuple (script name, language which the script was written, .rds output name)
    def __init__(self, data_manager:DataManager, fs_methods, aggregator, thresholds:list):

        self.dm = data_manager
        self.fs_methods = FSelector.generate_fselectors_object(fs_methods)
        self.aggregator = Aggregator(aggregator)
        self.rankings_to_aggregate = None

        self.thresholds = thresholds
        self.current_threshold = None



    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            
            self.dm.current_fold_iteration = i
            output_path = self.dm.get_output_path(fold_iteration=i)

            training_indexes, _ = self.dm.get_fold_data()
            training_data = self.dm.pd_df.iloc[training_indexes]
            
            rankings = []
            for fs_method in self.fs_methods:   
                rankings.append(
                    fs_method.select(training_data, output_path)
                )
                
            
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


    # just select the features looking at the whole dataset (no cross validation envolved)
    def select_features(self):
        
        Logger.whole_dataset_selection()
        output_path = self.dm.results_path + SELECTION_PATH

        rankings = []
        for fs_method in self.fs_methods:  
            rankings.append(
                fs_method.select(self.dm.pd_df, output_path)
            )
        
        self.__set_rankings_to_aggregate(rankings)
        file_path = output_path + AGGREGATED_RANKING_FILE_NAME
        for th in self.thresholds:
            Logger.aggregating_rankings()
            Logger.for_threshold(th)
            self.current_threshold = th
            aggregation = self.aggregator.aggregate(self)
            
            self.dm.save_encoded_ranking(aggregation, file_path+str(th)) 
        return
