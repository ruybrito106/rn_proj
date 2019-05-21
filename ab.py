import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

import pickle
from sklearn.externals import joblib

from project import Project
from structure_experiment import read_chunk_structured

# READ DATA

data = read_chunk_structured(index=2)
X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]

# TRAIN

clf = AdaBoostClassifier(n_estimators=256, learning_rate=0.6)
history = clf.fit(X_train, y_train)

# with open('out/ab_test_1.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# CLASSIFY

# clf2 = joblib.load('out/ab_test_1.pkl')
project = Project()
project.display_metrics(clf, X_test, y_test, mode=1)