import pandas as pd
import numpy as np
from ReliefF import ReliefF

def select(df):
    
    data = np.array(df.iloc[:, :-1])
    labels = np.array(df.iloc[:, -1:]).flatten()

    #k=10  # Kononenko, 1994
    k = len(labels)-1  # For the data we tested on, it was observed a k=10
                       # is not suitable and selects features with significantly
                       # lower predictive pontential (accuracy/ROC-AUC)


    fs = ReliefF(n_neighbors=k)
    fs.fit(data, labels)

    features = list(df.columns)
    data = {}
    data['features'] = []
    data['rank'] = []
    for i, feature in enumerate(fs.top_features):
        data['features'].append(features[feature])
        data['rank'].append(i+1)

    rank = pd.DataFrame(data, columns=['rank']).set_index(pd.Index(data['features']))

    return rank