import importlib
import os.path
import sys

class Aggregator:

    # heavy: tells whether to use the heavy selection method, i. e., for each fold
    #   iteration, build a dictionary containing all rankings from
    #   all fs methods and keep it in memory until the next fold iteration. This is
    #   useful if you want, for example, to measure the stability of the first layer
    #   rankings of each fs method(in the Hybrid ensemble, for example).

    # threshold_sensitive: tells whether the aggregation method could potentially
    # change its final aggregated ranking depending on the threshold value 
    def __init__(self, aggregation_method):
        
        self.aggregation_method = aggregation_method
        self.user_script = False
        self.__check_for_script_file()

        # Specified by each aggregator algorithm
        self.heavy = self.__is_heavy_required()
        self.threshold_sensitive = self.__is_threshold_sensitive()
        


    def __check_for_script_file(self):
        pkgdir = sys.modules['efsassembler'].__path__[0] + "/"
        user_alg_path = pkgdir + "aggregators/user_algorithms/"
        user_script = os.path.isfile(user_alg_path + self.aggregation_method + ".py")

        if user_script:
            self.user_script = True
        else:
            self.user_script = False
        return


    def __is_heavy_required(self):
        try:
            if self.user_script:
                return importlib.import_module("efsassembler.aggregators.user_algorithms" + \
                                                self.aggregation_method).heavy
            else:
                return importlib.import_module("efsassembler.aggregators." + \
                                                self.aggregation_method).heavy
        except:
            return False


    def __is_threshold_sensitive(self):
        try:
            if self.user_script:
                return importlib.import_module("efsassembler.aggregators.user_algorithms" + \
                                                self.aggregation_method).threshold_sensitive
            else:
                return importlib.import_module("efsassembler.aggregators." + \
                                                self.aggregation_method).threshold_sensitive
        except:
            return False


    # selector: a Hybrid/Heterogeneous/Homogeneous object
    def aggregate(self, selector): 
        if self.user_script:
            agg_foo = importlib.import_module("efsassembler.aggregators.user_algorithms" + \
                                                self.aggregation_method).aggregate
        else:
            agg_foo = importlib.import_module("efsassembler.aggregators." + \
                                                self.aggregation_method).aggregate
        
        return agg_foo(self, selector)

    