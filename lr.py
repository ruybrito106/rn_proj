import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle
from joblib import dump, load
from sklearn.externals import joblib

from project import Project
from structure_experiment import read_chunk_structured

# READ DATA

data = read_chunk_structured(index=2)
X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]

# TRAIN

clf = LogisticRegression(verbose=1)
history = clf.fit(X_train, y_train)
dump(clf, 'out/linear/lr.joblib')

# TEST

clf2 = load('out/linear/lr.joblib')
project = Project()
project.display_metrics(clf2, X_test, y_test, mode=1)