from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd

def select(df):
    X = df.iloc[0:, 0:-1]
    y = df.iloc[:, -1]

    estimator = SVR(kernel="linear")
    ranker = RFE(estimator=estimator, step=1)
    ranker = ranker.fit(X, y)


    data = {"rank": ranker.ranking_}
    rank = pd.DataFrame(data, index=list(X.columns))
    rank = rank.sort_values(by="rank")

    return rank

