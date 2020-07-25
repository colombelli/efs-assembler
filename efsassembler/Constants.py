MAX_SEED = 9999999
AGGREGATED_RANKING_FILE_NAME = "agg_ranking_th"

# Used in Evaluator and InformationManager
ROC_AUC_METRIC = "roc"
PRECISION_RECALL_AUC_METRIC = "pr" 
ACCURACY_METRIC = "acc"
FINAL_CONFUSION_MATRICES_FILE_NAME = "final_confusion_matrices.pkl"

# Used in InformationManager
SINGLE_FS_DESIGN = "Single FS Method"
HYBRID_FS_DESIGN = "Hybrid Ensemble"
HETEROGENEOUS_FS_DESIGN = "Heterogeneous Ensemble"
HOMOGENEOUS_FS_DESIGN = "Homogeneous Ensemble"

# info file
INFO_FILE_NAME = "experiment_info.txt"
TITLE_INFO_FILE = "Experiment Informations\n"
DESIGN_INFO_FILE = "Design: "
NUM_FOLDS_INFO_FILE = "Chosen k for stratified k-fold cv: "
NUM_BOOTSTRAPS_INFO_FILE = "Number of bootstraps: "
HYB_FST_AGG_INFO_FILE = "First aggregation method: "
HYB_SND_AGG_INFO_FILE = "Second aggregation method: "
HET_HOM_AGG_INFO_FILE = "Aggregation method: "
DATASET_USED = "Dataset: "
SINGLE_FS_INFO_FILE = "Feature Selection method: "
MULTIPLE_FS_INFO_FILE = "Feature Selection methods:"
SEED_INFO_FILE = "Seed: "

INTERMEDIATE_RESULTS_FOLDER_NAME = "intermediate_results"

# csv files
CSV_PRED_TABLE_FILE_NAME = "prediction_performance_results.csv"
CSV_AUC_TABLE_FILE_NAME = "_aucs_results.csv"
CSV_ACCURACY_TABLE_FILE_NAME = "accuracies_results.csv"
CSV_STB_TABLE_FILE_NAME = "stbs_results.csv"
CSV_FINAL_RESULTS_TABLE_FILE_NAME = "final_results.csv"
CSV_AUC_TABLE_COLUMNS = ["th_frac", "th_int"] #auc1, auc2, ..., aucn
CSV_FINAL_RESULTS_TABLE_COLUMNS = ["th_frac", "th_int", "stabilities", "mean_ACC", "std_ACC", "mean_ROC_AUC", "std_ROC_AUC", "mean_PR_AUC", "std_PR_AUC"]

LVL2_CSV_ACCURACY_TABLE_FILE_NAME = "lvl2_accs_results.csv"
LVL2_CSV_ROC_AUC_TABLE_FILE_NAME = "lvl2_roc_aucs_results.csv"
LVL2_CSV_PR_AUC_TABLE_FILE_NAME = "lvl2_pr_aucs_results.csv"
LVL2_CSV_STB_TABLE_FILE_NAME = "lvl2_stbs_results.csv"
LVL2_CSV_FINAL_RESULTS_TABLE_FILE_NAME = "lvl2_final_results.csv"
LVL2_CSV_FINAL_RESULTS_TABLE_COLUMNS = ["th_frac", "th_int", "mean_stb", "std_stb", "mean_ACC", "std_ACC", "mean_ROC_AUC", "std_ROC_AUC", "mean_PR_AUC", "std_PR_AUC"]