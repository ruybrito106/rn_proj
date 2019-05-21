from pandas import read_csv

from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV

from project import Project

# READ DATA

train, valid = read_csv('data/chunk_0/train.csv', sep='\t', header=0, index_col=0), read_csv('data/chunk_0/valid.csv', sep='\t', header=0, index_col=0)
test = read_csv('data/test.csv', sep='\t', header=0, index_col=0)

train_X, train_y = train.iloc[:, 1:-2], train.iloc[:, -2]
valid_X, valid_y = valid.iloc[:, 1:-2], valid.iloc[:, -2]
test_X, test_y = test.iloc[:, 1:-2], test.iloc[:, -2]

# CREATE MODEL

def init_model_fn():
    model = Sequential()
    model.add(Dense(32, activation='softplus', input_dim=243))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='softplus'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))

    adam = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=["accuracy"])
    return model

# RUN GRID SEARCH

param_grid = dict(batch_size=[64, 128, 256])
model = KerasClassifier(build_fn=init_model_fn, epochs=100000, verbose=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid.fit(train_X, train_y, callbacks=[EarlyStopping(patience=4)], validation_data=(valid_X, valid_y))

params = grid.best_params_

# CLASSIFY

classifier = init_model_fn()
history = classifier.fit(train_X, train_y, callbacks=[EarlyStopping(patience=25)], epochs=100000, batch_size=int(params['batch_size']), validation_data=(valid_X, valid_y))

project = Project()
project.plot_training_error_curves(history)
project.display_metrics(classifier, test_X, test_y, history)