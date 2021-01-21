from efsassembler.Logger import Logger
from efsassembler.Constants import SELECTION_PATH
from efsassembler.StratifiedKFold import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.utils import resample
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import random
from tensorflow import random as tf_random
from os import mkdir, environ
from sys import exit
import pickle
import urllib.parse
from copy import deepcopy


class DataManager:

    def __init__(self, results_path, file_path, num_bootstraps, 
                num_folds, undersampling=True, seed=0):
        
        self.seed = seed
        self.set_seed()

        self.file_path = file_path
        self.num_bootstraps = num_bootstraps
        self.num_folds = num_folds
        self.undersampling = undersampling
       
        #self.r_df = None
        self.pd_df = None
        self.__load_dataset()
        self.pd_df = self.pd_df.reset_index(drop=True)
        

        self.folds = None
        self.current_fold_iteration = 0
        self.current_bootstraps = None
        self.bs_rankings = None     # only used if heavy selection is required by 
                                    # any of the aggregation methods

        self.results_path = results_path
        self.folds_final_selection = None


    def set_seed(self):
        np.random.seed(self.seed)
        robjects.r['set.seed'](self.seed)

        # From https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
        environ['PYTHONHASHSEED']=str(self.seed)
        random.seed(self.seed)
        tf_random.set_seed(self.seed)
        return


    # Prevents column class being interpreted as string
    def __convert_class_col_to_numeric(self):
        self.pd_df["class"] = pd.to_numeric(self.pd_df["class"], downcast="integer")
        return


    def __load_dataset(self):

        Logger.loading_x_dataset(self.file_path)
        if self.file_path[-3:] == "rds":
            r_df = self.load_RDS(self.file_path)
            self.pd_df = self.r_to_pandas(r_df)
            self.__convert_class_col_to_numeric()
        
        elif self.file_path[-3:] == "csv":
            self.pd_df = self.load_csv(self.file_path)
            self.__convert_class_col_to_numeric()
            #self.r_df = self.pandas_to_r(self.pd_df)

        else:
            raise("Dataset format not accepted. Should be a .rds or .csv file.")
        
        return



    def create_results_dir(self):

        Logger.creating_results_directory_in_x_path(self.results_path)
        try:
            mkdir(self.results_path)
        except:
            """
            print("Given directory already created, files will be replaced.")
            print("Note that if there's any missing folder inside this existent one, the execution will fail.")
            if input("Input c to cancel or any other key to continue... ") == "c":
                exit()
            else: 
                return
            """
            #Throw error message
            exit()
        
        mkdir(self.results_path+SELECTION_PATH)

        for i in range(1, self.num_folds+1):
            fold_dir = self.results_path+"/fold_"+str(i)
            mkdir(fold_dir)

            for j in range(1, self.num_bootstraps+1):
                bs_dir = fold_dir + "/bootstrap_"+str(j)
                mkdir(bs_dir)



    @classmethod
    def load_RDS(self, file_path):
        
        read_RDS = robjects.r['readRDS']
        return read_RDS(file_path)


    @classmethod
    def load_csv(self, file_path):
        return pd.read_csv(file_path, index_col=0)


    @classmethod
    def pandas_to_r(self, df):
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pandas_df = robjects.conversion.py2rpy(df)
        return r_from_pandas_df


    @classmethod
    def r_to_pandas(self, df):
        with localconverter(robjects.default_converter + pandas2ri.converter):
                pandas_from_r_df = robjects.conversion.rpy2py(df)
        return pandas_from_r_df


    # The below alnum encode and decode methods were taken from a StackOverflow's 
    # topic answer and sligthly modified. Their original versions are available in:
    # https://stackoverflow.com/questions/32035520/how-to-encode-utf-8-strings-with-only-a-z-a-z-0-9-and-in-python
    @classmethod
    def alnum_encode(self, text):
        if text == "class":
            return text
        return "X" + urllib.parse.quote(text, safe='')\
            .replace('-', '%2d').replace('.', '%2e').replace('_', '%5f')\
            .replace('%', '_')

    @classmethod
    def alnum_decode(self, underscore_encoded):
        if underscore_encoded == "class":
            return underscore_encoded
        underscore_encoded = underscore_encoded[1:]
        return urllib.parse.unquote(underscore_encoded.replace('_','%'), errors='strict')


    # Important note: The method only encodes columns (the features)
    @classmethod
    def encode_df(self, df):
        
        columns = []
        for attribute in df.columns:
            columns.append(self.alnum_encode(attribute))

        df.columns = columns
        return df


    # rows is a boolean parameter telling wheter the method must also decode its rows;
    # it is basically used when decoding ranking-like dataframes (where features are indexes) 
    @classmethod
    def decode_df(self, df, rows:bool):

        if not rows:
            columns = []
            for attribute in df.columns:
                columns.append(self.alnum_decode(attribute))
            df.columns = columns
        
        if rows:
            indexes = []
            for ind in df.index:
                indexes.append(self.alnum_decode(ind))
            df.index = indexes

        return df


    @classmethod
    def save_encoded_ranking(self, ranking, file_name_and_dir):
        encoded_ranking = deepcopy(ranking)
        decoded_ranking = self.decode_df(encoded_ranking, True)

        #r_decoded_ranking = self.pandas_to_r(decoded_ranking)
        #robjects.r["saveRDS"](r_decoded_ranking, file_name_and_dir)
        decoded_ranking.to_csv(file_name_and_dir+".csv")
        return

    
    def encode_main_dm_df(self):
        self.pd_df = self.encode_df(self.pd_df)
        #self.r_df = self.pandas_to_r(self.pd_df)
        return
    
    def decode_main_dm_df(self):
        self.pd_df = self.decode_df(self.pd_df, False)
        #self.r_df = self.pandas_to_r(self.pd_df)
        return



    def init_data_folding_process(self):

        self.__calculate_folds()
        self.__save_folds()
        return


    def __calculate_folds(self):

        k = self.num_folds
        skf = StratifiedKFold(self.pd_df, "class", k, undersampling=self.undersampling)
        self.folds = list(skf.split())
        return

    
    def __save_folds(self):
        
        file = self.results_path + "fold_sampling.pkl"
        with open(file, 'wb') as f:
            pickle.dump(self.folds, f)
        return


    def update_bootstraps(self):
        self.current_bootstraps = self.__get_bootstraps()
        self.__save_bootstraps()


    # Output: A list of tuples containing n tuples representing the n 
    #           (bootstraps, out-of-bag) samples
    def __get_bootstraps(self):
        
        training_data = self.folds[self.current_fold_iteration][0]
        num_bs_samples = len(training_data)
        
        bootstraps_oob = []
        for _ in range(self.num_bootstraps):
            bootstrap = resample(training_data, replace=True, n_samples=num_bs_samples) #random_state is not being used since seed is being set globally
            oob = np.array([x for x in training_data if x not in bootstrap])
            bootstraps_oob.append((bootstrap, oob))

        return bootstraps_oob


    def __save_bootstraps(self):

        path = self.results_path + "fold_" + str(self.current_fold_iteration+1) + "/bootstrap_"
        for i, bootstrap in enumerate(self.current_bootstraps):
            file = path + str(i+1) + "/bootstrap_sampling.pkl" 
            with open(file, 'wb') as f:
                pickle.dump(bootstrap, f)
        return

    
    def update_bootstraps_outside_cross_validation(self, df, pkl_path):
        
        num_bs_samples = len(df)
        numeric_indexes = list(df.index.values)
        
        bootstraps_oob = []
        for _ in range(self.num_bootstraps):
            bootstrap = resample(numeric_indexes, replace=True, n_samples=num_bs_samples)
            oob = np.array([x for x in numeric_indexes if x not in bootstrap])
            bootstraps_oob.append((bootstrap, oob))
        
        self.current_bootstraps = bootstraps_oob
        self.__save_bootstraps_outside_cv(pkl_path)
        return 

    
    def __save_bootstraps_outside_cv(self, pkl_path):
        for i, bootstrap in enumerate(self.current_bootstraps):
            file = pkl_path + "bootstrap_sampling_" + str(i+1) + ".pkl" 
            with open(file, 'wb') as f:
                pickle.dump(bootstrap, f)
        return


    def get_output_path(self, fold_iteration=None, bootstrap_iteration=None):
        
        path = self.results_path

        if fold_iteration is None:
            return path
        
        path += "fold_" + str(fold_iteration+1) + "/"
        if bootstrap_iteration is None:
            return path
        
        path += "bootstrap_" + str(bootstrap_iteration+1) + "/"
        return path


    def get_fold_data(self):
        training_data = self.folds[self.current_fold_iteration][0]
        testing_data = self.folds[self.current_fold_iteration][1]
        return (training_data, testing_data)


    def set_bs_rankings(self, bs_rankings):
        self.bs_rankings = bs_rankings
        return

    
    def compute_data_folds_final_selection(self):
    
        label_counts = self.pd_df['class'].value_counts()
        majority_class = label_counts.idxmax()
        minority_class = 0 if majority_class else 1
        majority_indexes = self.pd_df.loc[self.pd_df["class"]==majority_class].index.tolist()
        minority_indexes = self.pd_df.loc[self.pd_df["class"]==minority_class].index.tolist()
        minority_count = label_counts.min()
        
        num_folds = round(label_counts[majority_class] / label_counts[minority_class])

        folds_final_selection = []
        prev_index = 0
        for _ in range(num_folds):
            new_fold = deepcopy(minority_indexes)
            limit_index = prev_index + minority_count
            
            if limit_index > len(majority_indexes):
                new_fold += majority_indexes[prev_index:]
            else:
                new_fold += majority_indexes[prev_index:limit_index]
            
            random.shuffle(new_fold)
            folds_final_selection.append(new_fold)
            prev_index = limit_index


        if limit_index < len(majority_indexes): # Then the 'round' floored and there are more samples
            folds_final_selection[-1] += majority_indexes[limit_index:]
            random.shuffle(folds_final_selection[-1])

        file = self.results_path + SELECTION_PATH + "folds.pkl"
        with open(file, 'wb') as f:
            pickle.dump(folds_final_selection, f)
        
        self.folds_final_selection = folds_final_selection
        return


    def create_balanced_selection_dirs(self):
        for i in range(len(self.folds_final_selection)):
            mkdir(self.results_path+SELECTION_PATH+str(i))
        return