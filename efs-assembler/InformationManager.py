from engine.DataManager import DataManager
from engine.Evaluator import Evaluator
from engine.Constants import *
import csv
from numpy import array as np_array
from numpy import mean as np_mean
from numpy import std as np_std
from copy import deepcopy
from os import mkdir

class InformationManager:

    # aggregators: if there are, then it must be a python list cointaining either one string 
    # or two (in case the hybrid design was chosen); if there aren't (single FS), keep it None
    # methods: a list containing one or more strings representing the fs methods used
    def __init__(self, data_manager:DataManager, evaluator:Evaluator,
                 methods:list, aggregators=None):

        self.dm = data_manager
        self.evaluator = evaluator
        self.aggregators = aggregators
        self.methods = methods
        
        self.info_txt_lines = None
        self.design = self.__find_design()

        self.thresholds = None
        self.stabilities = None
        self.aucs = None

        self.__create_initial_file_info()


    def __find_design(self):

        if not(self.aggregators):
            self.__get_single_fs_info()
            return SINGLE_FS_DESIGN

        elif len(self.aggregators) == 2:
            self.__get_hybrid_info()
            return HYBRID_FS_DESIGN

        elif self.dm.num_bootstraps == 0:
            self.__get_heterogeneous_info()
            return HETEROGENEOUS_FS_DESIGN

        else:
            self.__get_homogeneous_info()
            return HOMOGENEOUS_FS_DESIGN



    def __create_initial_file_info(self):
        
        title = TITLE_INFO_FILE
        dataset = DATASET_USED + self.dm.file_path
        seed = SEED_INFO_FILE + str(self.dm.seed)
        design = DESIGN_INFO_FILE + self.design
        num_folds = NUM_FOLDS_INFO_FILE + str(self.dm.num_folds)

        lines = [title, dataset, seed, design, num_folds]
        self.info_txt_lines = lines + self.info_txt_lines

        self.__save_info_file()
        return        


    def __get_single_fs_info(self):

        fs_method = SINGLE_FS_INFO_FILE + self.methods[0]
        self.info_txt_lines = [fs_method]
        return
    
    def __get_homogeneous_info(self):

        num_bootstraps = NUM_BOOTSTRAPS_INFO_FILE + str(self.dm.num_bootstraps)
        fs_method = SINGLE_FS_INFO_FILE + self.methods[0]
        aggregator = HET_HOM_AGG_INFO_FILE + self.aggregators[0]

        self.info_txt_lines = [num_bootstraps, fs_method, aggregator]
        return
    
    def __get_heterogeneous_info(self):
        methods_title = MULTIPLE_FS_INFO_FILE
        fs_methods = ["- " + method for method in self.methods]
        aggregator = HET_HOM_AGG_INFO_FILE + self.aggregators[0]

        self.info_txt_lines = [methods_title] + fs_methods + [aggregator]
        return
    
    def __get_hybrid_info(self):
        num_bootstraps = NUM_BOOTSTRAPS_INFO_FILE + str(self.dm.num_bootstraps)
        methods_title = MULTIPLE_FS_INFO_FILE
        fs_methods = ["- " + method for method in self.methods]
        fst_aggregator = HYB_FST_AGG_INFO_FILE + self.aggregators[0]
        snd_aggregator = HYB_SND_AGG_INFO_FILE + self.aggregators[1]

        self.info_txt_lines = [num_bootstraps, methods_title] + fs_methods + \
                                [fst_aggregator, snd_aggregator]
        return


    def __save_info_file(self):

        with open(self.dm.results_path+INFO_FILE_NAME, "w") as file:
            for line in self.info_txt_lines:
                file.write(line)
                file.write("\n")
        return


    
    def create_csv_tables(self):
        self.__create_csv_auc_table(ROC_AUC_METRIC)
        self.__create_csv_auc_table(PRECISION_RECALL_AUC_METRIC)
        self.__create_csv_accuracy_table()
        self.__create_csv_final_results()
        return
    


    def __create_csv_auc_table(self, curve=ROC_AUC_METRIC):
        
        file_name = curve + CSV_AUC_TABLE_FILE_NAME
        with open(self.dm.results_path+file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            
            columns = deepcopy(CSV_AUC_TABLE_COLUMNS)
            for i in range(self.dm.num_folds):
                columns.append("AUC_"+str(i+1))

            writer.writerow(columns)

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                
                aucs = []
                for auc in self.evaluator.prediction_performances[curve]:
                    aucs.append(auc[i])

                row = [frac_th, th] + aucs
                writer.writerow(row)
        return


    def __create_csv_accuracy_table(self):
        
        with open(self.dm.results_path+CSV_ACCURACY_TABLE_FILE_NAME, 'w', newline='') as file:
            writer = csv.writer(file)
            
            columns = deepcopy(CSV_AUC_TABLE_COLUMNS)
            for i in range(self.dm.num_folds):
                columns.append("ACC_"+str(i+1))

            writer.writerow(columns)

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                
                accuracies = []
                for acc in self.evaluator.prediction_performances[ACCURACY_METRIC]:
                    accuracies.append(acc[i])

                row = [frac_th, th] + accuracies
                writer.writerow(row)
        return
        

    def __create_csv_final_results(self):

        with open(self.dm.results_path+CSV_FINAL_RESULTS_TABLE_FILE_NAME, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(CSV_FINAL_RESULTS_TABLE_COLUMNS)

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                stability = self.evaluator.stabilities[i]
                
                accuracies = np_array(
                    self.evaluator.prediction_performances[ACCURACY_METRIC]).transpose()[i]
                mean_acc = np_mean(accuracies)
                std_acc = np_std(accuracies)

                roc_aucs = np_array(
                    self.evaluator.prediction_performances[ROC_AUC_METRIC]).transpose()[i]
                mean_roc_auc = np_mean(roc_aucs)
                std_roc_auc = np_std(roc_aucs)

                pr_aucs = np_array(
                    self.evaluator.prediction_performances[PRECISION_RECALL_AUC_METRIC]).transpose()[i]
                mean_pr_auc = np_mean(pr_aucs)
                std_pr_auc = np_std(pr_aucs)

                row = [frac_th, th, stability, 
                        mean_acc, std_acc,
                        mean_roc_auc, std_roc_auc,
                        mean_pr_auc, std_pr_auc]
                writer.writerow(row)
        return


    def create_intermediate_csv_tables(self, level1_evaluation, level2_evaluation):
        print("\nCreating intermediate results csv files...")
        self.__create_intermediate_results_folder()
        self.__create_level1_csv_tables(level1_evaluation)
        self.__create_level2_csv_tables(level2_evaluation[0], level2_evaluation[1])
        return


    def __create_intermediate_results_folder(self):
        fold_dir = self.dm.results_path + "/" + INTERMEDIATE_RESULTS_FOLDER_NAME
        try:
            mkdir(fold_dir)
        except:
            print("Impossible to create directory:", fold_dir)
            print("Either due to pre-path inexistence or because folder already exists.")
        return


    def __create_level1_csv_tables(self, level1_evaluation):
        
        for fs_method in level1_evaluation:
            stabilities = level1_evaluation[fs_method][0]
            prediction_performances = level1_evaluation[fs_method][1]
            
            accs = prediction_performances[ACCURACY_METRIC]
            roc_aucs = prediction_performances[ROC_AUC_METRIC]
            pr_aucs = prediction_performances[PRECISION_RECALL_AUC_METRIC]


            accs_table_file_name = fs_method + "_" + CSV_ACCURACY_TABLE_FILE_NAME
            self.__create_intermediate_csv_accuracy_table(
                accs, accs_table_file_name) 

            roc_aucs_table_file_name = fs_method + "_roc" + CSV_AUC_TABLE_FILE_NAME
            self.__create_intermediate_csv_auc_table(
                roc_aucs, roc_aucs_table_file_name, ROC_AUC_METRIC)

            pr_aucs_table_file_name = fs_method + "_pr" + CSV_AUC_TABLE_FILE_NAME
            self.__create_intermediate_csv_auc_table(
                pr_aucs, pr_aucs_table_file_name, PRECISION_RECALL_AUC_METRIC)

            stb_table_file_name = fs_method + "_" + CSV_STB_TABLE_FILE_NAME
            self.__create_intermediate_csv_stabilities_table(stabilities, stb_table_file_name)

            final_results_file_name = fs_method + "_" + CSV_FINAL_RESULTS_TABLE_FILE_NAME
            self.__create_intermediate_csv_final_results(prediction_performances, stabilities, final_results_file_name)
        return


    def __create_level2_csv_tables(self, stabilities, prediction_performances):
        
        self.__create_intermediate_csv_auc_table(
            prediction_performances[ROC_AUC_METRIC], 
            LVL2_CSV_ROC_AUC_TABLE_FILE_NAME,
            ROC_AUC_METRIC
        )
        
        self.__create_intermediate_csv_auc_table(
            prediction_performances[PRECISION_RECALL_AUC_METRIC], 
            LVL2_CSV_PR_AUC_TABLE_FILE_NAME, 
            PRECISION_RECALL_AUC_METRIC
        )

        self.__create_intermediate_csv_accuracy_table(
            prediction_performances[ACCURACY_METRIC],
            LVL2_CSV_ACCURACY_TABLE_FILE_NAME
        )

        self.__create_intermediate_csv_stabilities_table(
            stabilities, LVL2_CSV_STB_TABLE_FILE_NAME)

        self.__create_intermediate_csv_final_results(
            prediction_performances, 
            stabilities, 
            LVL2_CSV_FINAL_RESULTS_TABLE_FILE_NAME
        )
        return

    
    def __create_intermediate_csv_auc_table(self, aucs, table_name, curve=ROC_AUC_METRIC):
        
        csv_path = self.dm.results_path+"/"+INTERMEDIATE_RESULTS_FOLDER_NAME+"/"+table_name
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            columns = deepcopy(CSV_AUC_TABLE_COLUMNS)
            for i in range(len(aucs)):
                columns.append("AUC_"+str(i+1))

            writer.writerow(columns)

            aucs = np_array(aucs).transpose()

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                row = [frac_th, th] + list(aucs[i])

                writer.writerow(row)
        return


    def __create_intermediate_csv_accuracy_table(self, accs, table_name):
        
        csv_path = self.dm.results_path+"/"+INTERMEDIATE_RESULTS_FOLDER_NAME+"/"+table_name
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            columns = deepcopy(CSV_AUC_TABLE_COLUMNS)
            for i in range(len(accs)):
                columns.append("ACC_"+str(i+1))

            writer.writerow(columns)

            accs = np_array(accs).transpose()

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                row = [frac_th, th] + list(accs[i])

                writer.writerow(row)
        return

    
    def __create_intermediate_csv_stabilities_table(self, stabilities, table_name):
        
        csv_path = self.dm.results_path+"/"+INTERMEDIATE_RESULTS_FOLDER_NAME+"/"+table_name
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            columns = deepcopy(CSV_AUC_TABLE_COLUMNS)
            for i in range(len(stabilities)):
                columns.append("Stb_"+str(i+1))

            writer.writerow(columns)

            stabilities = np_array(stabilities).transpose()

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]
                row = [frac_th, th] + list(stabilities[i])
                writer.writerow(row)
        return


    def __create_intermediate_csv_final_results(self, prediction_performances, stabilities, table_name):

        stabilities = np_array(stabilities).transpose()

        accs = np_array(prediction_performances[ACCURACY_METRIC]).transpose()
        roc_aucs = np_array(prediction_performances[ROC_AUC_METRIC]).transpose()
        pr_aucs = np_array(prediction_performances[PRECISION_RECALL_AUC_METRIC]).transpose()

        csv_path = self.dm.results_path+"/"+INTERMEDIATE_RESULTS_FOLDER_NAME+"/"+table_name
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(LVL2_CSV_FINAL_RESULTS_TABLE_COLUMNS)

            for i, th in enumerate(self.evaluator.thresholds):
                frac_th = self.evaluator.frac_thresholds[i]

                mean_stb = np_mean(stabilities[i])
                std_stb = np_std(stabilities[i])
                
                mean_acc = np_mean(accs[i])
                std_acc = np_std(accs[i])

                mean_roc_auc = np_mean(roc_aucs[i])
                std_roc_auc = np_std(roc_aucs[i])

                mean_pr_auc = np_mean(pr_aucs[i])
                std_pr_auc = np_std(pr_aucs[i])

                row = [frac_th, th, mean_stb, std_stb, 
                    mean_acc, std_acc, 
                    mean_roc_auc, std_roc_auc,
                    mean_pr_auc, std_pr_auc
                ]
                writer.writerow(row)

        return