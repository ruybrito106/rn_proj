import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import roc_curve, auc

import pickle
from joblib import dump, load
from sklearn.externals import joblib

from project import Project
from structure_experiment import read_chunk_structured

def get_roc_auc(y, pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, pred)
    return auc(false_positive_rate, true_positive_rate)

# READ DATA

raw = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
X, y = raw.iloc[:, :-2], raw.iloc[:, -2]

data = read_chunk_structured(index=2)
X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]

lr_train = []
lr_test = []

for K in [30, 60, 90, 120, 150, 180, 210, 240]:
    selection = SelectKBest(k=K)
    selection.fit_transform(X, y)
    fs = selection.get_support()

    _X_train = X_train.iloc[:, fs]
    _X_test = X_test.iloc[:, fs]

    clf = LogisticRegression(verbose=0)
    clf.fit(_X_train, y_train)

    train_pred = clf.predict(_X_train)
    lr_train.append(get_roc_auc(y_train, train_pred))

    test_pred = clf.predict(_X_test)
    lr_test.append(get_roc_auc(y_test, test_pred))

    print (K)

print (lr_train)
print (lr_test)