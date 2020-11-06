from . import geode
import pandas as pd
import numpy as np

def select(df):
    

    features = list(df.columns[0:len(df.columns)-1])
    labels = list(np.array(df['class']).astype('int') + 1)
    mat = df.iloc[:, 0:len(df.columns)-1].transpose().to_numpy()

    chdir_res = geode.chdir(mat, labels, features, calculate_sig=0, nnull=100)

    data = {}
    data['features'] = []
    data['rank'] = []
    for i, feature in enumerate(chdir_res):
        data['features'].append(feature[1])
        data['rank'].append(i+1)

    rank = pd.DataFrame(data, columns=['rank']).set_index(pd.Index(data['features']))
    return rank