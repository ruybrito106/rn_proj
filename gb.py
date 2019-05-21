from pandas import read_csv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from project import Project

# READ DATA

train, valid = read_csv('data/chunk_2/train.csv', sep='\t', header=0, index_col=0), read_csv('data/chunk_1/valid.csv', sep='\t', header=0, index_col=0)
test = read_csv('data/test.csv', sep='\t', header=0, index_col=0)

train_X, train_y = train.iloc[:, 1:-2], train.iloc[:, -2]
valid_X, valid_y = valid.iloc[:, 1:-2], valid.iloc[:, -2]
test_X, test_y = test.iloc[:, 1:-2], test.iloc[:, -2]

# CREATE MODEL

# model = GradientBoostingClassifier(verbose=1)

# CLASSIFY

classifier = GradientBoostingClassifier(n_estimators=500, loss='exponential', verbose=1)
history = classifier.fit(train_X, train_y)

project = Project()
project.display_metrics(classifier, test_X, test_y, history, mode=1)