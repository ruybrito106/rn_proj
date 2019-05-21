import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

import pickle
from joblib import dump, load
from sklearn.externals import joblib

from project import Project
from structure_experiment import read_chunk_structured

# READ DATA

raw = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
X, y = raw.iloc[:, :-2], raw.iloc[:, -2]

selection = SelectKBest(k=120)
selection.fit_transform(X, y)
fs = selection.get_support()

# data = read_chunk_structured(index=2)
# X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
# X_train = X_train.iloc[:, fs]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]
X_test = X_test.iloc[:, fs]

# TRAIN

# clf = SVC(probability=True, C=10, gamma=100, kernel='rbf', verbose=1, cache_size=2000, shrinking=False)
# history = clf.fit(X_train, y_train)
# dump(clf, 'out/svc_2.joblib')

# CLASSIFY

# project = Project()
# project.display_metrics(clf, X_test, y_test, mode=1)

clf2 = load('out/svc_2.joblib')
project = Project()
project.display_metrics(clf2, X_test, y_test, mode=1)