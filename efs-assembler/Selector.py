from engine.DataManager import DataManager as dm
import rpy2.robjects as robjects
import importlib
import os

class FSelector:

    # ranking_name: the name of the csv ranking output produced by the algorithm;
    def __init__(self, ranking_name, script_name):

        self.ranking_name = ranking_name
        self.script_name = script_name

    
    @classmethod
    def generate_fselectors_object(self, methods):
        
        fs_methods = []
        for script, language, ranking_name in methods:
            if language == "python":
                fs_methods.append(
                    PySelector(ranking_name, script)
                )
            elif language == "r":
                fs_methods.append(
                    RSelector(ranking_name, script)
                )
        return fs_methods



class RSelector(FSelector):

    def select(self, dataframe, output_path):
        dataframe = dm.pandas_to_r(dataframe)

        this_file_path = os.path.dirname(__file__)
        call = this_file_path + "/fs_algorithms/" + self.script_name + ".r"
        robjects.r.source(call)

        ranking = robjects.r["select"](dataframe)
        ranking = dm.r_to_pandas(ranking)
        
        dm.save_encoded_ranking(ranking, output_path+self.ranking_name)

        robjects.r['rm']('select')
        return ranking


class PySelector(FSelector):

    def __init__(self, ranking_name, script_name):
        FSelector.__init__(self, ranking_name, script_name)
        self.py_selection = importlib.import_module("engine.fs_algorithms."+script_name).select

    def select(self, dataframe, output_path):
        ranking = self.py_selection(dataframe)
        dm.save_encoded_ranking(ranking, output_path+self.ranking_name)
        return ranking