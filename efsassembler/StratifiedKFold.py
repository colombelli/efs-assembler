import random
import pandas as pd
import numpy as np
from copy import deepcopy

class StratifiedKFold:
    
    def __init__(self, seed, dataframe, class_column_name, k, undersampling=True):
        random.seed(seed)
        
        self.df = dataframe
        self.class_coloumn_name = class_column_name
        self.k = k
        self.undersampling = undersampling

        
        self.classes = self.df[self.class_coloumn_name].unique()
        self.class_counts = self.df[self.class_coloumn_name].value_counts().to_dict()
        self.minority_count = self.class_counts[min(self.class_counts)]
        
        self.class_folds = {}
        self.__init_class_folds_dict()   # fold distribution separeted in class key
        self.folds = self.__get_folds()   # a list with indexes names in another list, one per fold
        self.__shuffle_each_fold()


    def __init_class_folds_dict(self):

        for df_class in self.classes:
            self.class_folds[df_class] = None

        
    def __get_folds(self):
        
        
        final_folds = [[] for _ in range(self.k)]
        for df_class in self.classes:
    
            class_indexes = self.df.loc[self.df[self.class_coloumn_name] == df_class].index.to_list()
            amount_per_fold = self.class_counts[df_class] // self.k
            
            random.shuffle(class_indexes)
            current_class_folds = [[] for _ in range(self.k)]

            for class_fold in current_class_folds:
                self.__get_samples(class_fold, class_indexes, amount_per_fold)

        
            self.__distribute_remaining_samples(amount_per_fold, current_class_folds, final_folds, class_indexes)
            self.class_folds[df_class] = current_class_folds
            self.__append_in_final_folds(final_folds, current_class_folds)

        return final_folds
        
            
    def __get_samples(self, class_fold, samples, amount):
        
        for _ in range(amount):
            class_fold.append(samples.pop())
        return
    
    
    def __distribute_remaining_samples(self, current_amount, current_folds, final_folds, class_indexes):
        
        len_folds = np.array([len(x)+current_amount for x in final_folds])
        while class_indexes:
            fold_with_less_samples = len_folds.argmin()
            current_folds[fold_with_less_samples].append(class_indexes.pop())
            len_folds[fold_with_less_samples] += 1
            
        return


    def __append_in_final_folds(self, final_folds, current_class_folds):
        
        for i, samples in enumerate(current_class_folds):
            final_folds[i] = final_folds[i] + samples
        return
    
    
    def __shuffle_each_fold(self):
        
        for fold in self.folds:
            random.shuffle(fold)
        return
    
    
    def split(self):
        
        for i, fold in enumerate(self.folds):
            
            test_set = fold

            if self.undersampling:
                train_set = self.__get_undersampled_train_set(i)
            else:
                train_set = [item for j,sublist in enumerate(self.folds) 
                                if j!=i for item in sublist]
            
            yield (train_set, test_set)


    def __get_undersampled_train_set(self, test_fold_idx):

        undersampled_train_set = []
        for i in range(len(self.folds)):

            if i == test_fold_idx:
                continue
            else:
                undersampled_train_set += self.__random_undersample_fold(i)
        return undersampled_train_set


    def __random_undersample_fold(self, fold_idx):
        
        minority_class = min(self.class_counts)
        minority_class_fold_count = len(self.class_folds[minority_class][fold_idx])
        
        samples_to_remove = []
        for df_class in self.classes:
            if df_class == minority_class:
                continue
            
            else:
                current_class_count = len(self.class_folds[df_class][fold_idx])
                amount_to_remove = current_class_count - minority_class_fold_count
                samples_to_remove += random.sample(
                    self.class_folds[df_class][fold_idx], amount_to_remove)
          
        undersampled_fold = deepcopy(self.folds[fold_idx])
        for sample in samples_to_remove:
            undersampled_fold.remove(sample)

        return undersampled_fold