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
        self.final_rankings_dict = {}
        self.__init_final_rankings_dict()


    def __init_final_rankings_dict(self):
        for th in self.thresholds:
            self.final_rankings_dict[th] = []
        return


    # In order to reuse Evaluator class, we need an AGGREGATED_RANKING_FILE_NAME+th
    # accessible inside each fold iteration folder, so we simply resave the only
    # ranking we have with the appropriate name

    def select_features_experiment(self):

        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            
            self.dm.current_fold_iteration = i
            output_path = self.dm.get_output_path(fold_iteration=i)

            training_indexes, _ = self.dm.get_fold_data()
            training_data = self.dm.pd_df.iloc[training_indexes]

            ranking = self.fs_method.select(training_data, output_path)

            output_path = self.dm.get_output_path(fold_iteration=i)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
            for th in self.thresholds:
                self.dm.save_encoded_ranking(ranking, file_path+str(th)) 
            
        return



    def select_features(self, balanced=True):

        if balanced:
            total_folds = len(self.dm.folds_final_selection)
            for i, fold in enumerate(self.dm.folds_final_selection):
                Logger.final_balanced_selection_iter(i, total_folds-1)
                df = self.dm.pd_df.loc[fold]
                ranking = self.fs_method.select(df, save_ranking=False)
                output_path = self.dm.results_path + SELECTION_PATH + str(i) + '/'
                file_path = output_path + AGGREGATED_RANKING_FILE_NAME
                
                for th in self.thresholds:
                    self.dm.save_encoded_ranking(ranking, file_path+str(th))
                    self.final_rankings_dict[th].append(ranking)

        else:
            Logger.whole_dataset_selection()
            output_path = self.dm.results_path + SELECTION_PATH
            ranking = self.fs_method.select(self.dm.pd_df, save_ranking=False)
            output_path = self.dm.results_path + SELECTION_PATH + str(i) + '/'
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
            
            for th in self.thresholds:
                    self.dm.save_encoded_ranking(ranking, file_path+str(th))
             
        return