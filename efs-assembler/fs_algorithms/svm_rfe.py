from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd

def select(df):
    X = df.iloc[0:, 0:-1]
    y = df.iloc[:, -1]

    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 1, step=1)
    print("Ranking features with SVM-RFE...")
    selector = selector.fit(X, y)


    print("Processing data...")
    data = {"rank": selector.ranking_}
    rank = pd.DataFrame(data, index=list(X.columns))
    rank = rank.sort_values(by="rank")

    return rank

