import pandas as pd
from copy import deepcopy

import efsassembler.kuncheva_index as ki
from efsassembler import Hybrid


heavy = True    # Requires access to dm.bs_rankings
threshold_sensitive = True  # The threshold value may change the final aggregated ranking

def aggregate(self, selector:Hybrid):
    
    bs_rankings = selector.dm.bs_rankings
    threshold = selector.current_threshold


    fs_stabilities = get_fs_stabilities(threshold, bs_rankings)
    
    aggregated_ranking_base = initialize_aggregated_ranking_dict(bs_rankings)

    final_rankings = []    
    for bs in bs_rankings:
        aggregated_ranking = deepcopy(aggregated_ranking_base)

        for fs, ranking in enumerate(bs_rankings[bs]):
            reversed_ranking = ranking.iloc[::-1]
            for feature in reversed_ranking.index: 
                aggregated_ranking[feature] += (reversed_ranking.index.get_loc(feature)+1) * (fs_stabilities[fs]**5)

        final_rankings.append(
            build_df_and_correct_order(aggregated_ranking)
        )

    return final_rankings


def get_normalize_stability(stability):

    s_max = 1
    s_min = -1
    normalized = (stability - s_min) / (s_max - s_min) 

    return normalized*2



def initialize_aggregated_ranking_dict(bs_rankings):
    aggregated_ranking = {}
    for feature in list(bs_rankings[0][0].index):
        aggregated_ranking[feature] = 0
    return aggregated_ranking


def get_fs_stabilities(threshold, bs_rankings):
    
    num_fs = len(bs_rankings[0])
    
    stabilities = []
    for fs in range(num_fs):
        
        fs_rankings = []
        for bs in bs_rankings:
            df_ranking = bs_rankings[bs][fs]
            lst_features = list(df_ranking.index.values)
            fs_rankings.append(lst_features)
        
        normalized_stb = get_normalize_stability(ki.get_kuncheva_index(fs_rankings, threshold=threshold))
        stabilities.append(normalized_stb)  
    return stabilities


def build_df_and_correct_order(aggregated_ranking):
    final_ranking = pd.DataFrame.from_dict(aggregated_ranking, orient='index')
    final_ranking.columns = ['rank']
    final_ranking = final_ranking.sort_values(by='rank', ascending=False)
    final_ranking.iloc[:] = final_ranking.iloc[::-1].values
    return final_ranking
