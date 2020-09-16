from efsassembler.Logger import Logger

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
        self.experiment.select_features()
        self.dm.create_selection_dirs()
        self.experiment.select_features(self.balanced)
        self.aggregate_rankings()
        return


    def create_selection_dir(self):
        if self.balanced:
            self.dm.create_balanced_selection_dirs()
        return


    def aggregate_rankings(self):
        #load every th ranking
        #aggregate them for every 
        return
        