from efsassembler.Logger import Logger
from efsassembler.FeatureRanker import FeatureRanker
from efsassembler.DataManager import DataManager
from efsassembler.Aggregator import Aggregator

class FSTechnique:

    '''

        fr_methods: even if the technique uses only one feature ranker algorithm, 
        you need to provide it inside a list. The elements of that list are tuples
        with the following format:
        (script name, language which the script was written, .csv output name)

        An FSTechnique can have one aggregator (Homogeneous and Heterogeneous EFS), and
        in this case the 'aggregator' parameter must be used; two aggregators (Hybrid 
        EFS), and in this case 'fst_aggregator' and 'snd_aggregator' must be used; or
        none aggregator (Single FR type experiment), and in this case no aggregator
        related parameter shall be used.
        
    '''

    def __init__(self, data_manager:DataManager, fr_methods:list, thresholds:list,
                    aggregator=None, fst_aggregator=None, snd_aggregator=None):
        self.dm = data_manager
        self.thresholds = thresholds
        self.fr_methods = FeatureRanker.generate_ranker_object(fr_methods)
        self.fr_method = self.fr_methods[0]  # For techniques that use only one Feature Ranker 

        self.aggregator = None
        self.fst_aggregator = None
        self.snd_aggregator = None
        self.__init_aggregators(aggregator, fst_aggregator, snd_aggregator)
        
        self.threshold_sensitive = None
        self.__infer_if_threshold_sensitive()

        self.current_threshold = None
        self.rankings_to_aggregate = None

        self.final_rankings_dict = {}
        self.__init_final_rankings_dict()


    def _set_rankings_to_aggregate(self, rankings):
        self.rankings_to_aggregate = rankings
        return


    def __init_final_rankings_dict(self):
        for th in self.thresholds:
            self.final_rankings_dict[th] = []
        self.final_rankings_dict[0] = []
        return


    def __init_aggregators(self, aggregator, fst_aggregator, snd_aggregator):

        if aggregator and (fst_aggregator or snd_aggregator):
            raise Exception('Wrong type of experiment, see documentation for how to use aggregators parameters.')
        
        elif aggregator:
            self.aggregator = Aggregator(aggregator)
        
        elif fst_aggregator and snd_aggregator:
            self.fst_aggregator = Aggregator(fst_aggregator)
            self.snd_aggregator = Aggregator(snd_aggregator)
        
        elif (not aggregator) and (not fst_aggregator) and (not snd_aggregator):
            pass

        else:
            raise Exception('Wrong type of experiment, see documentation for how to use aggregators parameters.')
        return


    def __infer_if_threshold_sensitive(self):
        if (self.aggregator):
            self.threshold_sensitive = self.aggregator.threshold_sensitive
        
        elif (self.fst_aggregator):
            if  (self.fst_aggregator.threshold_sensitive) or \
                (self.snd_aggregator.threshold_sensitive):
                self.threshold_sensitive = True
            else:
                self.threshold_sensitive = False

        else:
            self.threshold_sensitive = False
        return


    
    '''
        Methods implemented specific by each specialized class
    '''
    def select_features(self, balanced=True):
        pass
        return

    def select_features_experiment(self):
        pass
        return