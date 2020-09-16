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


    def hom_feature_selection(self, output_path, i=None, experiment_selection=True):
        rankings = []
        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            
            ranking_path=None
            if i:
                Logger.bootstrap_fold_iteration(j+1, i+1)
                ranking_path = self.dm.get_output_path(i, j)
            
            else:
                Logger.bootstrap_iteration(j+1)
                
            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
            rankings.append(self.fs_method.select(bootstrap_data, ranking_path, save_ranking=experiment_selection))

        self.__set_rankings_to_aggregate(rankings)
        
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


    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()
            output_path = self.dm.get_output_path(fold_iteration=i)

            self.hom_feature_selection(output_path, i=i, experiment_selection=True)
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
        self.dm.update_bootstraps_outside_cross_validation(self.dm.pd_df, output_path)
        self.hom_feature_selection(output_path, experiment_selection=False)
        return


    def select_features_balanced(self):
        total_folds = len(self.dm.folds_final_selection)
        for i, fold in enumerate(self.dm.folds_final_selection):
            Logger.final_balanced_selection_iter(i, total_folds)
            output_path = self.dm.results_path + SELECTION_PATH + str(i) + '/'
            df = self.dm.pd_df.loc[fold]
            self.dm.update_bootstraps_outside_cross_validation(df, output_path)
            self.hom_feature_selection(output_path, experiment_selection=False)
        return