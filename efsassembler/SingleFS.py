from efsassembler.Logger import Logger
from efsassembler.Selector import FSelector, PySelector, RSelector
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME, SELECTION_PATH

class SingleFS:
    
    # fs_method: a single elemet list (to maintain coherence) whose element is a tuple: 
    # (script name, language which the script was written, .csv output name)
    def __init__(self, data_manager:DataManager, fs_method, thresholds:list):

        self.dm = data_manager
        self.fs_method = FSelector.generate_fselectors_object(fs_method)[0]
        self.thresholds = thresholds



    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            
            self.dm.current_fold_iteration = i
            output_path = self.dm.get_output_path(fold_iteration=i)

            training_indexes, _ = self.dm.get_fold_data()
            training_data = self.dm.pd_df.iloc[training_indexes]

            ranking = self.fs_method.select(training_data, output_path)

            # in order to reuse Evaluator class, we need an AGGREGATED_RANKING_FILE_NAME+th
            # accessible inside each fold iteration folder, so we simply resave the only
            # ranking we have with the appropriate name
            output_path = self.dm.get_output_path(fold_iteration=i)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
            for th in self.thresholds:
                self.dm.save_encoded_ranking(ranking, file_path+str(th)) 
            
        return



    def select_features(self):

        Logger.whole_dataset_selection()
        output_path = self.dm.results_path + SELECTION_PATH

        self.fs_method.select(self.dm.pd_df, output_path)
        return