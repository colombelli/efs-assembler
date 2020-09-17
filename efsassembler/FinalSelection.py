from efsassembler.Logger import Logger
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME

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
        self.rankings_dict = None


    def start(self):
        Logger.encoding_dataset()
        self.dm.encode_main_dm_df()
        self.dm.compute_data_folds_final_selection()
        self.create_selection_dirs()
        self.rankings_dict = self.experiment.select_features(self.balanced)
        self.aggregate_rankings()
        return


    def create_selection_dirs(self):
        if self.balanced:
            self.dm.create_balanced_selection_dirs()
        return


    def aggregate_rankings(self):

        #self.load_rankings()
        #aggregate them for every 
        return

    '''
    def load_rankings(self):

        root_path = self.dm.results_path + SELECTION_PATH
        for i in range(len(self.dm.folds_final_selection))
            for th in self.experiment.thresholds:
                ranking_path = root_path + str(i) + '/' + \
                    AGGREGATED_RANKING_FILE_NAME + str(th)
                ranking = self.dm.load_csv(ranking_path)
                try:
                    self.rankings_dict[th].append(ranking)
                except:
                    self.rankings_dict[th] = [ranking]
        return
    '''