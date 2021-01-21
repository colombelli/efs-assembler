from efsassembler.FSTechnique import FSTechnique
from efsassembler.Logger import Logger
from efsassembler.Constants import AGGREGATED_RANK_FILE_NAME, SINGLE_RANK_FILE_NAME, SELECTION_PATH
from efsassembler.Aggregator import Aggregator

class FinalSelection:

    '''
        Select the features looking at the whole dataset (no cross validation) or
        at multiple folds of the data considering, for each fold, all the samples 
        from the minority class and an equivalent amount for the majority class 
        (except, possibly, for the last fold)
    '''

    # balanced: if True, uses the folding fashion for balanced feature selection
    # with all minority class samples present into every fold
    def __init__(self, selection_technique:FSTechnique, balanced=True):

        self.balanced = balanced
        self.selection_technique = selection_technique
        self.dm = self.selection_technique.dm


    def start(self, skip_encoding=False):
        Logger.encoding_dataset()
        if not skip_encoding:
            self.dm.encode_main_dm_df()
        self.dm.compute_data_folds_final_selection()
        self.create_selection_dirs()
        self.selection_technique.select_features(self.balanced)
        if self.balanced:
            self.aggregate_rankings()
        return


    def create_selection_dirs(self):
        if self.balanced:
            self.dm.create_balanced_selection_dirs()
        return


    def aggregate_rankings(self):
        aggregator = Aggregator('borda')

        if self.selection_technique.threshold_sensitive:
            output_path = self.dm.results_path + SELECTION_PATH + AGGREGATED_RANK_FILE_NAME
            for th in self.selection_technique.thresholds:
                self.selection_technique.rankings_to_aggregate = self.selection_technique.final_rankings_dict[th]
                aggregation = aggregator.aggregate(self.selection_technique)
                file_name = output_path + str(th)
                self.dm.save_encoded_ranking(aggregation, file_name)
        else:
            file_name = self.dm.results_path + SELECTION_PATH + SINGLE_RANK_FILE_NAME
            self.selection_technique.rankings_to_aggregate = self.selection_technique.final_rankings_dict[0]
            aggregation = aggregator.aggregate(self.selection_technique)
            self.dm.save_encoded_ranking(aggregation, file_name)

        return