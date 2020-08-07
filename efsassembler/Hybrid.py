from efsassembler.Logger import Logger
from efsassembler.Selector import PySelector, RSelector
from efsassembler.Aggregator import Aggregator
from efsassembler.DataManager import DataManager
from efsassembler.Constants import AGGREGATED_RANKING_FILE_NAME, SELECTION_PATH

class Hybrid:
    
    # fs_methods: a tuple (script name, language which the script was written, .rds output name)
    # thresholds: must be list with integer values
    def __init__(self, data_manager:DataManager, fs_methods, 
    first_aggregator, second_aggregator, thresholds:list):

        self.dm = data_manager
        self.thresholds = thresholds
        self.current_threshold = None

        self.fs_methods = self.__generate_fselectors_object(fs_methods)
        self.fst_aggregator = Aggregator(first_aggregator)
        self.snd_aggregator = Aggregator(second_aggregator)

        if self.fst_aggregator.heavy or self.snd_aggregator.heavy:
            self.select_features_experiment = self.select_features_heavy_experiment
            self.select_features = self.select_features_heavy
        else:
            self.select_features_experiment = self.select_features_light_experiment
            self.select_features = self.select_features_light

        self.rankings_to_aggregate = None
            


        
    def __generate_fselectors_object(self, methods):
        
        fs_methods = []
        for script, language, rds_name in methods:
            if language == "python":
                fs_methods.append(
                    PySelector(rds_name, script)
                )
            elif language == "r":
                fs_methods.append(
                    RSelector(rds_name, script)
                )

        return fs_methods



    def select_features_light_experiment(self):
    
        for i in range(self.dm.num_folds):
            
            Logger.fold_iteration(i+1)
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()
             
            # initializes snd_layer_ranking
            snd_layer_rankings = {}
            for th in self.thresholds:
                snd_layer_rankings[th] = []

            for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
                Logger.bootstrap_fold_iteration(j+1, i+1)
                output_path = self.dm.get_output_path(i, j)
                bootstrap_data = self.dm.pd_df.iloc[bootstrap]

        
                fst_layer_rankings = []
                for fs_method in self.fs_methods:  
                    fst_layer_rankings.append(
                        fs_method.select(bootstrap_data, output_path)
                    )
                
                self.__set_rankings_to_aggregate(fst_layer_rankings)
                self.__aggregate_light_fst_layer(j, snd_layer_rankings)
            

            self.__aggregate_light_snd_layer(snd_layer_rankings)

        return


    def __aggregate_light_fst_layer(self, bootstrap_num, snd_layer_rankings, in_experiment=True):
        
        fold_iteration = self.dm.current_fold_iteration

        if in_experiment:
            output_path = self.dm.get_output_path(fold_iteration, bootstrap_num)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME
        

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
        fold_iteration = self.dm.current_fold_iteration

        if in_experiment:
            output_path = self.dm.get_output_path(fold_iteration=fold_iteration)
            file_path = output_path + AGGREGATED_RANKING_FILE_NAME

        else:
            file_path = self.dm.results_path + SELECTION_PATH + AGGREGATED_RANKING_FILE_NAME

        for th in self.thresholds:
            Logger.for_threshold(th)
            Logger.aggregating_n_level_rankings(2)
            self.current_threshold = th
            self.__set_rankings_to_aggregate(snd_layer_rankings[th])
            fs_aggregation = self.snd_aggregator.aggregate(self)
            self.dm.save_encoded_ranking(fs_aggregation, file_path+str(th))

        return


    def select_features_heavy_experiment(self):
    
        for i in range(self.dm.num_folds):
            
            Logger.fold_iteration(i+1)
            self.dm.current_fold_iteration = i
            self.dm.update_bootstraps()

            bs_rankings = {}
            for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
                Logger.bootstrap_fold_iteration(j+1, i+1)
                output_path = self.dm.get_output_path(i, j)
                bootstrap_data = self.dm.pd_df.iloc[bootstrap]

        
                fst_layer_rankings = []
                for fs_method in self.fs_methods: 
                    fst_layer_rankings.append(
                        fs_method.select(bootstrap_data, output_path)
                    )
                
                bs_rankings[j] = fst_layer_rankings
            self.dm.set_bs_rankings(bs_rankings)


            self.__aggregate_heavy() 
        return


    def __aggregate_heavy(self, in_experiment=True):

        fold_iteration = self.dm.current_fold_iteration
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
                    output_path = self.dm.get_output_path(fold_iteration, bs_num)
                    file_path = output_path + AGGREGATED_RANKING_FILE_NAME + str(th)
                    self.dm.save_encoded_ranking(fs_aggregation, file_path)
            
            snd_layer_rankings = fs_aggregations
            Logger.aggregating_n_level_rankings(2)

            if in_experiment:
                file_path = self.dm.get_output_path(fold_iteration=fold_iteration) + \
                                AGGREGATED_RANKING_FILE_NAME + str(th) 
            else:
                file_path = self.dm.results_path + SELECTION_PATH + \
                        AGGREGATED_RANKING_FILE_NAME + str(th) 

            self.__set_rankings_to_aggregate(snd_layer_rankings)
            final_ranking = self.snd_aggregator.aggregate(self)
            self.dm.save_encoded_ranking(final_ranking, file_path)
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

    
    def select_features_light(self):
    
        Logger.whole_dataset_selection()
        self.dm.update_bootstraps_outside_cross_validation()
            
        # initializes snd_layer_ranking
        snd_layer_rankings = {}
        for th in self.thresholds:
            snd_layer_rankings[th] = []

        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            Logger.bootstrap_iteration(j+1)
            bootstrap_data = self.dm.pd_df.iloc[bootstrap]

    
            fst_layer_rankings = []
            for fs_method in self.fs_methods:   
                fst_layer_rankings.append(
                    fs_method.select(bootstrap_data, save_ranking=False)
                )
            
            self.__set_rankings_to_aggregate(fst_layer_rankings)
            self.__aggregate_light_fst_layer(j, snd_layer_rankings, in_experiment=False)
        

        self.__aggregate_light_snd_layer(snd_layer_rankings, in_experiment=False)
        return

    
    def select_features_heavy(self):

        Logger.whole_dataset_selection()
        self.dm.update_bootstraps_outside_cross_validation()
            
        bs_rankings = {}
        for j, (bootstrap, _) in enumerate(self.dm.current_bootstraps):
            Logger.bootstrap_iteration(j+1)
            bootstrap_data = self.dm.pd_df.iloc[bootstrap]
    
            fst_layer_rankings = []
            for fs_method in self.fs_methods:   
                fst_layer_rankings.append(
                    fs_method.select(bootstrap_data, save_ranking=False)
                )
            
            bs_rankings[j] = fst_layer_rankings
        self.dm.set_bs_rankings(bs_rankings)

        self.__aggregate_heavy(in_experiment=False) 
        return


    # method for reusing the selection results of previous selection proccess
    # also useful for getting the performance of different thresholds used in stability
    # wheightened aggregation (which, if has more than one th, uses the mean th value)
    def post_aggregation(self):
        return