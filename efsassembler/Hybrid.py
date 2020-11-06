from efsassembler.Logger import Logger
from efsassembler.FeatureRanker import PyRanker, RRanker
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANK_FILE_NAME, SELECTION_PATH

class Hybrid:
    
    # fr_methods: a tuple (script name, language which the script was written, .rds output name)
    # thresholds: must be list with integer values
    def __init__(self, data_manager:DataManager, fr_methods, 
    first_aggregator, second_aggregator, thresholds:list):

        self.dm = data_manager
        self.thresholds = thresholds
        self.current_threshold = None

        self.fr_methods = self.__generate_ranker_object(fr_methods)
        self.fst_aggregator = Aggregator(first_aggregator)
        self.snd_aggregator = Aggregator(second_aggregator)

        if self.fst_aggregator.heavy or self.snd_aggregator.heavy:
            self.hyb_feature_selection = self.hyb_feature_selection_heavy
        else:
            self.hyb_feature_selection = self.hyb_feature_selection_light

        self.rankings_to_aggregate = None
        self.final_rankings_dict = {}
        self.__init_final_rankings_dict()
            

        
    def __generate_ranker_object(self, methods):
        
        fr_methods = []
        for script, language, rds_name in methods:
            if language == "python":
                fr_methods.append(
                    PyRanker(rds_name, script)
                )
            elif language == "r":
                fr_methods.append(
                    RRanker(rds_name, script)
                )
        return fr_methods


    def __init_final_rankings_dict(self):
        for th in self.thresholds:
            self.final_rankings_dict[th] = []
        return


#################################################################################################################################
##################################################### LIGHT SELECTION ###########################################################
#################################################################################################################################

    def hyb_feature_selection_light(self, in_experiment=True):
        
        ranking_path = None
        snd_layer_rankings = {}
        i = self.dm.current_fold_iteration

        for th in self.thresholds:
            snd_layer_rankings[th] = []

        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):

            if in_experiment:
                Logger.bootstrap_fold_iteration(j+1, i+1)
                ranking_path = self.dm.get_output_path(i, j)

            else:
                Logger.bootstrap_iteration(j+1)

            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
            fst_layer_rankings = []
            for fr_method in self.fr_methods:   
                fst_layer_rankings.append(
                    fr_method.select(bootstrap_data, ranking_path, save_ranking=in_experiment)
                )
            
            self.__set_rankings_to_aggregate(fst_layer_rankings)
            self.__aggregate_light_fst_layer(j, snd_layer_rankings, in_experiment)
        
        self.__aggregate_light_snd_layer(snd_layer_rankings, in_experiment)
        return 


    def __aggregate_light_fst_layer(self, bootstrap_num, snd_layer_rankings, in_experiment=True):
        
        fold_iteration = self.dm.current_fold_iteration

        if in_experiment:
            output_path = self.dm.get_output_path(fold_iteration, bootstrap_num)
            file_path = output_path + AGGREGATED_RANK_FILE_NAME
        

        for th in self.thresholds:
            Logger.for_threshold(th)
            Logger.aggregating_n_level_rankings(1)
            self.current_threshold = th
            fs_aggregation = self.fst_aggregator.aggregate(self)
            if in_experiment:
                self.dm.save_encoded_ranking(fs_aggregation, file_path+str(th))
            snd_layer_rankings[th].append(fs_aggregation)

        return

    
    def __aggregate_light_snd_layer(self, snd_layer_rankings, in_experiment=True):
        i = self.dm.current_fold_iteration

        if in_experiment:
            output_path = self.dm.get_output_path(fold_iteration=i)
            file_path = output_path + AGGREGATED_RANK_FILE_NAME

        elif i != None: # Final Selection balanced
            file_path = self.dm.results_path + SELECTION_PATH + str(i) + "/" + AGGREGATED_RANK_FILE_NAME
        
        else:  # Final Selection on the whole dataset
            file_path = self.dm.results_path + SELECTION_PATH + AGGREGATED_RANK_FILE_NAME

        for th in self.thresholds:
            Logger.for_threshold(th)
            Logger.aggregating_n_level_rankings(2)
            self.current_threshold = th
            self.__set_rankings_to_aggregate(snd_layer_rankings[th])
            fs_aggregation = self.snd_aggregator.aggregate(self)
            self.dm.save_encoded_ranking(fs_aggregation, file_path+str(th))

            if (not in_experiment) and (i != None):
                self.final_rankings_dict[th].append(fs_aggregation)

        return


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################



#################################################################################################################################
##################################################### HEAVY SELECTION ###########################################################
#################################################################################################################################

    def hyb_feature_selection_heavy(self, in_experiment=True):

        ranking_path = None
        bs_rankings = {}
        i = self.dm.current_fold_iteration

        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            
            if in_experiment:
                Logger.bootstrap_fold_iteration(j+1, i+1)
                ranking_path = self.dm.get_output_path(i, j)

            else: 
                Logger.bootstrap_iteration(j+1)

            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
            fst_layer_rankings = []
            for fr_method in self.fr_methods: 
                fst_layer_rankings.append(
                    fr_method.select(bootstrap_data, ranking_path, in_experiment)
                )
            
            bs_rankings[j] = fst_layer_rankings
        self.dm.set_bs_rankings(bs_rankings)

        self.__aggregate_heavy(in_experiment) 
        return


    def __aggregate_heavy(self, in_experiment=True):

        i = self.dm.current_fold_iteration
        for th in self.thresholds:
            Logger.for_threshold(th)
            Logger.aggregating_n_level_rankings(1)
            self.current_threshold = th

            if self.fst_aggregator.heavy:
                fs_aggregations = self.fst_aggregator.aggregate(self)
            else:
                fs_aggregations = self.__fst_aggregate_not_heavy(th)

            if in_experiment: 
                for bs_num, fs_aggregation in enumerate(fs_aggregations):
                    output_path = self.dm.get_output_path(i, bs_num)
                    file_path = output_path + AGGREGATED_RANK_FILE_NAME + str(th)
                    self.dm.save_encoded_ranking(fs_aggregation, file_path)
            
            snd_layer_rankings = fs_aggregations
            Logger.aggregating_n_level_rankings(2)

            if in_experiment:
                file_path = self.dm.get_output_path(fold_iteration=i) + \
                                AGGREGATED_RANK_FILE_NAME + str(th) 
           
            elif i != None:
                file_path = self.dm.results_path + SELECTION_PATH + str(i) + "/" + \
                        AGGREGATED_RANK_FILE_NAME + str(th) 

            else:
                file_path = self.dm.results_path + SELECTION_PATH + \
                        AGGREGATED_RANK_FILE_NAME + str(th) 

            self.__set_rankings_to_aggregate(snd_layer_rankings)
            final_ranking = self.snd_aggregator.aggregate(self)
            self.dm.save_encoded_ranking(final_ranking, file_path)

            if (not in_experiment) and (i != None):
                self.final_rankings_dict[th].append(final_ranking)
        return


    def __fst_aggregate_not_heavy(self, th):

        fs_aggregations = []
        for bs in self.dm.bs_rankings:
            rankings = self.dm.bs_rankings[bs]
            self.__set_rankings_to_aggregate(rankings)
            aggregation = self.fst_aggregator.aggregate(self)
            fs_aggregations.append(aggregation)
        
        return fs_aggregations


    def __set_rankings_to_aggregate(self, rankings):
        self.rankings_to_aggregate = rankings
        return


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################


    def select_features(self, balanced=True):

        if balanced:
            total_folds = len(self.dm.folds_final_selection)
            for i, fold in enumerate(self.dm.folds_final_selection):
                Logger.final_balanced_selection_iter(i, total_folds-1)
                self.dm.current_fold_iteration = i
                df = self.dm.pd_df.loc[fold]
                output_path = self.dm.results_path + SELECTION_PATH + str(i) + '/'
                self.dm.update_bootstraps_outside_cross_validation(df, output_path)
                self.hyb_feature_selection(in_experiment=False)
        else:
            Logger.whole_dataset_selection()
            self.dm.current_fold_iteration = None
            output_path = self.dm.results_path + SELECTION_PATH
            self.dm.update_bootstraps_outside_cross_validation(self.dm.pd_df, output_path)
            self.hyb_feature_selection(in_experiment=False)
        return


    def select_features_experiment(self):
    
        for i in range(self.dm.num_folds):
            Logger.fold_iteration(i+1)
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()
            self.hyb_feature_selection(in_experiment=True)
        return


    # method for reusing the selection results of previous selection proccess
    # also useful for getting the performance of different thresholds used in stability
    # wheightened aggregation (which, if has more than one th, uses the mean th value)
    def post_aggregation(self):
        return