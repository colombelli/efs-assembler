from . import geode
import pandas as pd
import numpy as np

def select(df):
    

    genes = list(df.columns[0:len(df.columns)-1])
    labels = list(np.array(df['class']).astype('int') + 1)
    mat = df.iloc[:, 0:len(df.columns)-1].transpose().to_numpy()

    print("Ranking features with Characteristic Direction (GeoDE)...")
    chdir_res = geode.chdir(mat, labels, genes, calculate_sig=0, nnull=100)

    print("Processing data...")
    data = {}
    data['gene'] = []
    data['rank'] = []
    for i, gene in enumerate(chdir_res):
        data['gene'].append(gene[1])
        data['rank'].append(i+1)

    rank = pd.DataFrame(data, columns=['rank']).set_index(pd.Index(data['gene']))
    return rank