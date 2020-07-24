from engine.Selector import FSelector, PySelector, RSelector
from engine.Aggregator import Aggregator
from engine.DataManager import DataManager
from engine.Constants import AGGREGATED_RANKING_FILE_NAME

class Heterogeneous:
    
    # fs_methods: a tuple (script name, language which the script was written, .rds output name)
    def __init__(self, data_manager:DataManager, fs_methods, aggregator, thresholds:list):

        self.dm = data_manager
        self.fs_methods = FSelector.generate_fselectors_object(fs_methods)
        self.aggregator = Aggregator(aggregator)
        self.rankings_to_aggregate = None

        self.thresholds = thresholds
        self.current_threshold = None



    def select_features(self):

        for i in range(self.dm.num_folds):
            print("\n\n################# Fold iteration:", i+1, "#################")
            
            self.dm.current_fold_iteration = i
            output_path = self.dm.get_output_path(fold_iteration=i)

            training_indexes, _ = self.dm.get_fold_data()
            training_data = self.dm.pd_df.loc[training_indexes]
            
            rankings = []
            for fs_method in self.fs_methods:   
                print("")
                rankings.append(
                    fs_method.select(training_data, output_path)
                )
                
            
            self.__set_rankings_to_aggregate(rankings)
            output_path = self.dm.get_output_path(fold_iteration=i)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
            for th in self.thresholds:
                print("\nAggregating rankings...")
                print("\n\nThreshold:", th)
                self.current_threshold = th
                aggregation = self.aggregator.aggregate(self)
                
                self.dm.save_encoded_ranking(aggregation, file_path+str(th)) 
        return


    def __set_rankings_to_aggregate(self, rankings):
        self.rankings_to_aggregate = rankings
        return
