from pandas import read_csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from project import Project

# READ DATA

train, valid = read_csv('data/chunk_1/train.csv', sep='\t', header=0, index_col=0), read_csv('data/chunk_1/valid.csv', sep='\t', header=0, index_col=0)
test = read_csv('data/test.csv', sep='\t', header=0, index_col=0)

train_X, train_y = train.iloc[:, 1:-2], train.iloc[:, -2]
valid_X, valid_y = valid.iloc[:, 1:-2], valid.iloc[:, -2]
test_X, test_y = test.iloc[:, 1:-2], test.iloc[:, -2]

# CREATE MODEL

model = RandomForestClassifier(verbose=1)

# RUN GRID SEARCH

# param_grid = {
#     'n_estimators': [300, 400, 500],
#     'max_depth': [40, 50]
# }

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
# grid.fit(train_X, train_y)

# print (grid.best_params_)

# CLASSIFY

# classifier = RandomForestClassifier(n_estimators=int(grid.best_params_['n_estimators']), max_depth=int(grid.best_params_['max_depth']), verbose=1)
classifier = RandomForestClassifier(n_estimators=500, max_depth=30, verbose=1)
history = classifier.fit(train_X, train_y)

project = Project()
project.display_metrics(classifier, test_X, test_y, history, mode=1)