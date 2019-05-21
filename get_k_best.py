import pandas as pd
from sklearn.feature_selection import SelectKBest

def selectKBest(K=115):
    data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
    X, y = data.iloc[:, :-2], data.iloc[:, -2]

    best = SelectKBest(k=K)
    best.fit_transform(X, y)

    return data.iloc[:, best.get_support()]