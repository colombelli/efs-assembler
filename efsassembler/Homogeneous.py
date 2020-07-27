from efsassembler.Selector import FSelector, PySelector, RSelector
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME

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
            print("\n\n################# Fold iteration:", i+1, "#################")
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()

            rankings = []
            for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
                print("\n\nBootstrap: ", j+1, "| Fold iteration:", i+1, "\n")
                
                output_path = self.dm.get_output_path(i, j)
                bootstrap_data = self.dm.pd_df.iloc[bootstrap]
                rankings.append(self.fs_method.select(bootstrap_data, output_path))

                
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

"""
    def select_features(self):

        print("Selecting features using the whole dataset...")

        rankings = []
        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            print("\n\nBootstrap: ", j+1, "\n")
            
            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
            rankings.append(self.fs_method.select(bootstrap_data, output_path))

            
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
"""