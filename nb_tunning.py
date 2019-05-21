import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import roc_curve, auc

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]

    while len(cls_1) < len(cls_0):
        cls_1 = cls_1.append(cls_1)

    return cls_0.append(cls_1.iloc[:len(cls_0)])

def get_roc_auc(y, pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, pred)
    return auc(false_positive_rate, true_positive_rate)

# READ DATA

data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=100)
data = balance_data(data)

X, y = data.iloc[:, 1:-2], data.iloc[:, -2]

# NUMBER FEATURES

import numpy as np
options = np.linspace(5, 240, 48)

fs_train = []
fs_test = []

for x in options:
    _X = SelectKBest(k=int(x)).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.25, random_state=100)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    fs_train.append(get_roc_auc(y_train, train_pred))

    test_pred = clf.predict(X_test)
    fs_test.append(get_roc_auc(y_test, test_pred))

    print (x)

print (fs_train)
print (fs_test)