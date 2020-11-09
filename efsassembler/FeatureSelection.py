from efsassembler.Logger import Logger
from efsassembler.FeatureRanker import FeatureRanker
from efsassembler.DataManager import DataManager

class FeatureSelection:

    # fr_method: even if the technique uses only one feature ranker algorithm, 
    # you need to provide it inside a list. The elements of that list are tuples:
    # (script name, language which the script was written, .csv output name)
    def __init__(self, data_manager:DataManager, fr_methods:list, thresholds:list):
        self.dm = data_manager
        self.thresholds = thresholds
        self.fr_methods = FeatureRanker.generate_ranker_object(fr_methods)

        self.final_rankings_dict = {}
        self.__init_final_rankings_dict()


    def __init_final_rankings_dict(self):
        for th in self.thresholds:
            self.final_rankings_dict[th] = []
        self.final_rankings_dict[0] = []
        return

    
    def select_features(self, balanced=True):
        pass
        return

    def select_features_experiment(self):
        pass
        return

