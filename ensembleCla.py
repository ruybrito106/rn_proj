from pandas import read_csv

import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoosting, ClassifierVotingClassifier
from sklearn.model_selection import GridSearchCV

from project import Project

# READ DATA

train, valid = read_csv('data/chunk_3/train.csv', sep='\t', header=0, index_col=0), read_csv('data/chunk_1/valid.csv', sep='\t', header=0, index_col=0)
test = read_csv('data/test.csv', sep='\t', header=0, index_col=0)

train_X, train_y = train.iloc[:, 1:-2], train.iloc[:, -2]
valid_X, valid_y = valid.iloc[:, 1:-2], valid.iloc[:, -2]
test_X, test_y = test.iloc[:, 1:-2], test.iloc[:, -2]

# CREATE MODEL

model = ensembleCla(probability=True, verbose=1)

# RUN GRID SEARCH

# param_grid = {
#     'kernel': ['rbf', 'poly'],
#     'C': np.logspace(0.01, 1000, 2),
#     'gamma': np.logspace(0.01, 1000, 2)
# }

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
# grid.fit(train_X, train_y)

# print (grid.best_params_)

# CLASSIFY

# classifier = SVC(probability=True, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel=grid.best_params_['kernel'], verbose=1)
rf = RandomForestClassifier(n_estimators=500, max_depth=30, verbose=1)
gb = GradientBoostingClassifier(n_estimators=500, loss='exponential', verbose=1)
svc = SVC(probability=True, gamma=100, kernel='rbf', verbose=1)


classifier = VotingClassifier(estimators=[ ('rf', rf), ('gb', gb), ('svc', svc)], weights=[1,1,1],flatten_transform=True)
history = classifier.fit(train_X, train_y)

project = Project()
project.display_metrics(classifier, test_X, test_y, history, mode=1)
