import pandas as pd
import numpy as np
from ReliefF import ReliefF

def select(df):
    
    data = np.array(df.iloc[0:, 0:len(df.columns)-1])
    labels = np.array(df.iloc[0:, len(df.columns)-1:]).flatten()

    fs = ReliefF(n_neighbors=len(labels)-1)
    print("Ranking features with ReliefF algorithm...")
    fs.fit(data, labels)

    print("Processing data...")
    genes = list(df.columns)
    data = {}
    data['gene'] = []
    data['rank'] = []
    for i, gene in enumerate(fs.top_features):
        data['gene'].append(genes[gene])
        data['rank'].append(i+1)

    rank = pd.DataFrame(data, columns=['rank']).set_index(pd.Index(data['gene']))

    return rank