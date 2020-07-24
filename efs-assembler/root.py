from engine.DataManager import DataManager
from engine.Hybrid import Hybrid
from engine.Heterogeneous import Heterogeneous
from engine.Homogeneous import Homogeneous
from engine.SingleFS import SingleFS
from engine.Evaluator import Evaluator
from engine.InformationManager import InformationManager
import rpy2.robjects.packages as rpackages
from time import time


def compute_print_time(st):
    
    print("\n\nTIME TAKEN:")
    end = time()
    try:
        hours, rem = divmod(end-st, 3600)
        minutes, seconds = divmod(rem, 60)
 
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    except:
        print(end-st)
    return



rpackages.quiet_require('FSelectorRcpp')
rpackages.quiet_require('FSelector')


num_bootstraps = 50
num_folds = 5

fs_methods = [
    ("reliefF", "python", "rf"),
    ("geoDE", "python", "gd"),
    ("gain-ratio", "r", "gr"),
    ("symmetrical-uncertainty", "r", "su"),
    ("oneR", "r", "or")
]

ths = [1, 5, 10, 15, 25, 50, 75, 100, 150, 200]
seed = 42

str_methods = ["ReliefF", "GeoDE", "Gain Ratio", "Symmetrical Uncertainty", "OneR"]



def perform_selection_hyb(dataset_path, results_path, aggregator1, aggregator2):
    
    str_aggregators = [aggregator1, aggregator2]

    dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, seed)
    dm.encode_main_dm_df()
    dm.create_results_dir()
    dm.init_data_folding_process()

    ev = Evaluator(dm, ths, False)
    im = InformationManager(dm, ev, str_methods, str_aggregators)
    ensemble = Hybrid(dm, fs_methods, aggregator1, aggregator2, ths)

    st = time()
    ensemble.select_features()
    compute_print_time(st)

    print("\n\nDecoding dataframe...")
    dm.decode_main_dm_df()
    print("\nStarting evaluation process...")
    ev.evaluate_final_rankings()

    print("\n\nCreating csv files...")
    im.create_csv_tables()

    print("\nEvaluating inner levels...")
    level1_evaluation, level2_evaluation = ev.evaluate_intermediate_hyb_rankings()

    print("\n\nCreating csv files...")
    im.create_intermediate_csv_tables(level1_evaluation, level2_evaluation)

    print("\nDone!\n\n")
    print("#################################################################\n")
    return



def perform_selection_het(dataset_path, results_path, aggregator):

    str_aggregators = [aggregator]

    num_bootstraps = 0

    dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, seed)
    dm.encode_main_dm_df()
    dm.create_results_dir()
    dm.init_data_folding_process()
    
    ev = Evaluator(dm, ths, False)
    im = InformationManager(dm, ev, str_methods, str_aggregators)
    ensemble = Heterogeneous(dm, fs_methods, aggregator, ths)

    st = time()
    ensemble.select_features()
    compute_print_time(st)

    print("\n\nDecoding dataframe...")
    dm.decode_main_dm_df()
    print("\nStarting evaluation process...")
    ev.evaluate_final_rankings()

    print("\n\nCreating csv files...")
    im.create_csv_tables()

    print("\nDone!\n\n")
    print("#################################################################\n")
    return



def perform_selection_hom(dataset_path, results_path, fs_method, aggregator):

    str_aggregators = [aggregator]

    dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, seed)
    dm.encode_main_dm_df()
    dm.create_results_dir()
    dm.init_data_folding_process()

    ev = Evaluator(dm, ths, False)
    im = InformationManager(dm, ev, str_methods, str_aggregators)
    ensemble = Homogeneous(dm, fs_method, aggregator, ths)

    st = time()
    ensemble.select_features() 
    compute_print_time(st)

    print("\n\nDecoding dataframe...")
    dm.decode_main_dm_df()
    print("\nStarting evaluation process...")
    ev.evaluate_final_rankings()

    print("\n\nCreating csv files...")
    im.create_csv_tables()

    print("\nDone!\n\n")
    print("#################################################################\n")
    return



def perform_selection_single(dataset_path, results_path, fs_method):

    str_aggregators = ["No aggregation"] 

    num_bootstraps = 0

    dm = DataManager(results_path, dataset_path, num_bootstraps, num_folds, seed)
    dm.encode_main_dm_df()
    dm.create_results_dir()
    dm.init_data_folding_process()

    ev = Evaluator(dm, ths, False)
    im = InformationManager(dm, ev, str_methods, str_aggregators)
    feature_selector = SingleFS(dm, fs_method, ths)

    st = time()
    feature_selector.select_features()
    compute_print_time(st)

    print("\n\nDecoding dataframe...")
    dm.decode_main_dm_df()
    print("\nStarting evaluation process...")
    ev.evaluate_final_rankings()

    print("\n\nCreating csv files...")
    im.create_csv_tables()

    print("\nDone!\n\n")
    print("#################################################################\n")
    return



def run():

    aggregator1 = "borda"
    aggregator2 = "stb_weightened_layer1"

    ########### HYBRID EXPERIMENTS ##############

    dataset_path = "/home/colombelli/Documents/datasets/research/kirp.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hyb_stb_borda/"
    perform_selection_hyb(dataset_path, results_path, aggregator2, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/ucec.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hyb_stb_borda/"
    perform_selection_hyb(dataset_path, results_path, aggregator2, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/thca.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hyb_stb_borda/"
    perform_selection_hyb(dataset_path, results_path, aggregator2, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/brca.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hyb_stb_borda/"
    perform_selection_hyb(dataset_path, results_path, aggregator2, aggregator1)

    """

    ########### HETEROGENOUS EXPERIMENTS ##############

    dataset_path = "/home/colombelli/Documents/datasets/research/kirp.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Het_borda/"
    perform_selection_het(dataset_path, results_path, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/ucec.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Het_borda/"
    perform_selection_het(dataset_path, results_path, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/thca.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Het_borda/"
    perform_selection_het(dataset_path, results_path, aggregator1)

    dataset_path = "/home/colombelli/Documents/datasets/research/brca.rds"
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Het_borda/"
    perform_selection_het(dataset_path, results_path, aggregator1)




    ########### HOMOGENEOUS EXPERIMENTS ##############

    method_relieff = [("reliefF", "python", "rf")]
    method_geode = [("geoDE", "python", "gd")]
    method_gr = [("gain-ratio", "r", "gr")]
    method_su = [("symmetrical-uncertainty", "r", "su")]
    method_oner = [("oneR", "r", "or")]

    ############ KIRP HOMOGENEOUS ############
    dataset_path = "/home/colombelli/Documents/datasets/research/kirp.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hom_borda_gr/"
    perform_selection_hom(dataset_path, results_path, method_gr, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hom_borda_su/"
    perform_selection_hom(dataset_path, results_path, method_su, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hom_borda_geode/"
    perform_selection_hom(dataset_path, results_path, method_geode, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hom_borda_relieff/"
    perform_selection_hom(dataset_path, results_path, method_relieff, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/Hom_borda_oner/"
    perform_selection_hom(dataset_path, results_path, method_oner, aggregator1)



    ############ UCEC HOMOGENEOUS ############
    dataset_path = "/home/colombelli/Documents/datasets/research/ucec.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hom_borda_gr/"
    perform_selection_hom(dataset_path, results_path, method_gr, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hom_borda_su/"
    perform_selection_hom(dataset_path, results_path, method_su, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hom_borda_geode/"
    perform_selection_hom(dataset_path, results_path, method_geode, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hom_borda_relieff/"
    perform_selection_hom(dataset_path, results_path, method_relieff, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/Hom_borda_oner/"
    perform_selection_hom(dataset_path, results_path, method_oner, aggregator1)



    ############ THCA HOMOGENEOUS ############
    dataset_path = "/home/colombelli/Documents/datasets/research/thca.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hom_borda_gr/"
    perform_selection_hom(dataset_path, results_path, method_gr, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hom_borda_su/"
    perform_selection_hom(dataset_path, results_path, method_su, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hom_borda_geode/"
    perform_selection_hom(dataset_path, results_path, method_geode, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hom_borda_relieff/"
    perform_selection_hom(dataset_path, results_path, method_relieff, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/Hom_borda_oner/"
    perform_selection_hom(dataset_path, results_path, method_oner, aggregator1)



    ############ BRCA HOMOGENEOUS ############
    dataset_path = "/home/colombelli/Documents/datasets/research/brca.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hom_borda_gr/"
    perform_selection_hom(dataset_path, results_path, method_gr, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hom_borda_su/"
    perform_selection_hom(dataset_path, results_path, method_su, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hom_borda_geode/"
    perform_selection_hom(dataset_path, results_path, method_geode, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hom_borda_relieff/"
    perform_selection_hom(dataset_path, results_path, method_relieff, aggregator1)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/Hom_borda_oner/"
    perform_selection_hom(dataset_path, results_path, method_oner, aggregator1)





    ########### SINGLE FS EXPERIMENTS ##############


    ######### KIRP
    dataset_path = "/home/colombelli/Documents/datasets/research/kirp.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/sin_gr/"
    perform_selection_single(dataset_path, results_path, method_gr)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/sin_su/"
    perform_selection_single(dataset_path, results_path, method_su)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/sin_geode/"
    perform_selection_single(dataset_path, results_path, method_geode)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/sin_relieff/"
    perform_selection_single(dataset_path, results_path, method_relieff)
    results_path = "/home/colombelli/Documents/Experiments11_mai/KIRP/sin_oner/"
    perform_selection_single(dataset_path, results_path, method_oner)


    ######### UCEC
    dataset_path = "/home/colombelli/Documents/datasets/research/ucec.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/sin_gr/"
    perform_selection_single(dataset_path, results_path, method_gr)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/sin_su/"
    perform_selection_single(dataset_path, results_path, method_su)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/sin_geode/"
    perform_selection_single(dataset_path, results_path, method_geode)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/sin_relieff/"
    perform_selection_single(dataset_path, results_path, method_relieff)
    results_path = "/home/colombelli/Documents/Experiments11_mai/UCEC/sin_oner/"
    perform_selection_single(dataset_path, results_path, method_oner)



    ######### THCA
    dataset_path = "/home/colombelli/Documents/datasets/research/thca.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/sin_gr/"
    perform_selection_single(dataset_path, results_path, method_gr)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/sin_su/"
    perform_selection_single(dataset_path, results_path, method_su)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/sin_geode/"
    perform_selection_single(dataset_path, results_path, method_geode)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/sin_relieff/"
    perform_selection_single(dataset_path, results_path, method_relieff)
    results_path = "/home/colombelli/Documents/Experiments11_mai/THCA/sin_oner/"
    perform_selection_single(dataset_path, results_path, method_oner)


    ######### BRCA
    dataset_path = "/home/colombelli/Documents/datasets/research/brca.rds"

    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/sin_gr/"
    perform_selection_single(dataset_path, results_path, method_gr)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/sin_su/"
    perform_selection_single(dataset_path, results_path, method_su)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/sin_geode/"
    perform_selection_single(dataset_path, results_path, method_geode)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/sin_relieff/"
    perform_selection_single(dataset_path, results_path, method_relieff)
    results_path = "/home/colombelli/Documents/Experiments11_mai/BRCA/sin_oner/"
    perform_selection_single(dataset_path, results_path, method_oner)
    """