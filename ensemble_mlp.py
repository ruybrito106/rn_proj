from pandas import read_csv

import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV

from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from project import Project

# READ DATA

train, valid = read_csv('data/chunk_4/train.csv', sep='\t', header=0, index_col=0), read_csv('data/chunk_3/valid.csv', sep='\t', header=0, index_col=0)
test = read_csv('data/test.csv', sep='\t', header=0, index_col=0)

train_X, train_y = train.iloc[:, 1:-2], train.iloc[:, -2]
valid_X, valid_y = valid.iloc[:, 1:-2], valid.iloc[:, -2]
test_X, test_y = test.iloc[:, 1:-2], test.iloc[:, -2]

# CREATE MODEL


# CLASSIFY

# MLP 1
mlp_1 = KerasClassifier(build_fn=create_sklearn_compatible_model, epochs=100000, verbose=1)
mlp_1.fit(train_X, train_y, callbacks=[EarlyStopping(patience=3)], validation_data=(valid_X, valid_y))

#MLP 2
mlp_2 = KerasClassifier(build_fn=create_sklearn_compatible_model, epochs=100000, verbose=1)
mlp_2.fit(train_X, train_y, callbacks=[EarlyStopping(patience=3)], validation_data=(valid_X, valid_y))

#MLP 3
mlp_3 = KerasClassifier(build_fn=create_sklearn_compatible_model, epochs=100000, verbose=1)
mlp_3.fit(train_X, train_y, callbacks=[EarlyStopping(patience=3)], validation_data=(valid_X, valid_y))

# ENSEMBLE
ensemble_mlp = EnsembleVoteClassifier(clfs=[mlp_1, mlp_2, mlp_3], weights=[1,1,1], voting='soft', refit=False)
history = ensemble_mlp.fit(train_X, train_y)

project = Project()
project.display_metrics(ensemble_mlp, test_X, test_y, history)
