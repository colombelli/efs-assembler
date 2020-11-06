import pandas as pd

heavy = False    # Doesn't requires access to dm.bs_rankings
threshold_sensitive = False  # The threshold value doesn't change the final aggregated ranking

def aggregate(self, selector):

    rankings = selector.rankings_to_aggregate

    aggregated_ranking = {}  # it's a dictionary where the keys 
                            # represent the features and its values 
                            # are, at first, the sum of the ranking
                            # positions and, by the end, the accumulative
                            # value of the rankings 

    for feature in rankings[0].index:
        aggregated_ranking[feature] = 0

    for ranking in rankings:
        for feature in ranking.index: 
            aggregated_ranking[feature] += ranking.index.get_loc(feature)+1 #this way 0=>1


    final_ranking = pd.DataFrame.from_dict(aggregated_ranking, orient='index')
    final_ranking.columns = ['rank']
    
    return final_ranking.sort_values(by='rank')

