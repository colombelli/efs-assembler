from efsassembler.Logger import Logger
from efsassembler.FeatureRanker import FeatureRanker, PyRanker, RRanker
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANK_FILE_NAME, SINGLE_RANK_FILE_NAME, SELECTION_PATH

class Heterogeneous:
    
    # fr_methods: a tuple (script name, language which the script was written, .rds output name)
    def __init__(self, data_manager:DataManager, fr_methods, aggregator, thresholds:list):

        self.dm = data_manager
        self.fr_methods = FeatureRanker.generate_ranker_object(fr_methods)
        self.aggregator = Aggregator(aggregator)
        self.rankings_to_aggregate = None

        self.thresholds = thresholds
        self.current_threshold = None
        self.final_rankings_dict = {}
        self.__init_final_rankings_dict()


    def __init_final_rankings_dict(self):
        for th in self.thresholds:
            self.final_rankings_dict[th] = []
        self.final_rankings_dict[0] = []
        return


    def het_feature_selection(self, df, output_path, in_experiment=True):
        
        rankings = []
        for fr_method in self.fr_methods:  
            rankings.append(
                fr_method.select(df, output_path, save_ranking=in_experiment)
            )
        
        self.__set_rankings_to_aggregate(rankings)

        if self.aggregator.threshold_sensitive:
            file_path = output_path + AGGREGATED_RANK_FILE_NAME
            for th in self.thresholds:
                Logger.aggregating_rankings()
                Logger.for_threshold(th)
                self.current_threshold = th
                aggregation = self.aggregator.aggregate(self)
                self.dm.save_encoded_ranking(aggregation, file_path+str(th)) 
                if not in_experiment:
                    self.final_rankings_dict[th].append(aggregation)

        else:
            file_path = output_path + SINGLE_RANK_FILE_NAME
            Logger.aggregating_rankings()
            aggregation = self.aggregator.aggregate(self)
            self.dm.save_encoded_ranking(aggregation, file_path)
            if not in_experiment:
                self.final_rankings_dict[0].append(aggregation)

        return


    def __set_rankings_to_aggregate(self, rankings):
        self.rankings_to_aggregate = rankings
        return



    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            
            self.dm.current_fold_iteration = i
            output_path = self.dm.get_output_path(fold_iteration=i)

            training_indexes, _ = self.dm.get_fold_data()
            training_data = self.dm.pd_df.iloc[training_indexes]
            
            self.het_feature_selection(training_data, output_path, in_experiment=True)

        return


    def select_features(self, balanced=True):
        if balanced:
            self.select_features_balanced()
        else:
            self.select_features_whole_dataset()
        return


    def select_features_whole_dataset(self):
        Logger.whole_dataset_selection()
        output_path = self.dm.results_path + SELECTION_PATH
        self.het_feature_selection(self.dm.pd_df, output_path, in_experiment=False)
        return


    def select_features_balanced(self):
        total_folds = len(self.dm.folds_final_selection)
        for i, fold in enumerate(self.dm.folds_final_selection):
            Logger.final_balanced_selection_iter(i, total_folds-1)
            output_path = self.dm.results_path + SELECTION_PATH + str(i) + '/'
            df = self.dm.pd_df.loc[fold]
            self.het_feature_selection(df, output_path, in_experiment=False)
        return
