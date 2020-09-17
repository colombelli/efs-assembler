from efsassembler.Logger import Logger
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME, SELECTION_PATH
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
    # experiment: a Hybrid/Heterogeneous/Homogeneous/SingleFS object
    def __init__(self, experiment, datamanager, balanced=True):

        self.balanced = balanced
        self.experiment = experiment
        self.dm = datamanager


    def start(self):
        Logger.encoding_dataset()
        self.dm.encode_main_dm_df()
        self.dm.compute_data_folds_final_selection()
        self.create_selection_dirs()
        self.experiment.select_features(self.balanced)
        self.aggregate_rankings()
        return


    def create_selection_dirs(self):
        if self.balanced:
            self.dm.create_balanced_selection_dirs()
        return


    def aggregate_rankings(self):
        aggregator = Aggregator('borda')
        output_path = self.dm.results_path + SELECTION_PATH + AGGREGATED_RANKING_FILE_NAME
        for th in self.experiment.thresholds:
            self.experiment.rankings_to_aggregate = self.experiment.final_rankings_dict[th]
            aggregation = aggregator.aggregate(self.experiment)
            file_name = output_path + str(th)
            self.dm.save_encoded_ranking(aggregation, file_name)

        return