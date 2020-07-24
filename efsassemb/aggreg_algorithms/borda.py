import pandas as pd


def aggregate(self, selector):

        rankings = selector.rankings_to_aggregate

        aggregated_ranking = {}  # it's a dictionary where the keys 
                                # represent the genes and its values 
                                # are, at first, the sum of the ranking
                                # positions and, by the end, the accumulative
                                # value of the rankings 


        for gene in rankings[0].index:
            aggregated_ranking[gene] = 0

        for ranking in rankings:
            for gene in ranking.index: 
                aggregated_ranking[gene] += ranking.index.get_loc(gene)+1 #this way 0=>1


        final_ranking = pd.DataFrame.from_dict(aggregated_ranking, orient='index')
        final_ranking.columns = ['rank']
       
        return final_ranking.sort_values(by='rank')

