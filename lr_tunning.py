import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest

import pickle
from joblib import dump, load
from sklearn.externals import joblib

from project import Project
from structure_experiment import read_chunk_structured

# READ DATA

raw = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
X, y = raw.iloc[:, :-2], raw.iloc[:, -2]

selection = SelectKBest(k=180)
selection.fit_transform(X, y)
fs = selection.get_support()

data = read_chunk_structured(index=2)
X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
X_train = X_train.iloc[:, fs]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]
X_test = X_test.iloc[:, fs]

# GRID

grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100], 
    "penalty": ["l1", "l2"],
}

clf = LogisticRegression(verbose=1)
clf = GridSearchCV(clf, grid, cv=5, verbose=1)
clf.fit(X_train, y_train)

params = clf.best_params_
print("tuned hpyerparameters: (best parameters) ", params)
print("accuracy: ",clf.best_score_)

# TRAIN

clf = LogisticRegression(C=params["C"], penalty=params["penalty"], verbose=1)
history = clf.fit(X_train, y_train)
dump(clf, 'out/lr_2.joblib')

# CLASSIFY

clf2 = load('out/lr_2.joblib')
project = Project()
project.display_metrics(clf2, X_test, y_test, mode=1)