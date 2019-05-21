import pandas as pd

import pickle
from joblib import dump, load

from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest

from project import Project
from structure_experiment import read_ensemble_structured

def balance_ensemble_data(data):
    cls_0 = data[data['value'] == 0]
    cls_1 = data[data['value'] == 1]
    if len(cls_1) < len(cls_0):
        while len(cls_1) < len(cls_0):
            cls_1 = cls_1.append(cls_1)
        
        return cls_0.append(cls_1.iloc[:len(cls_0)])
    else:
        while len(cls_0) < len(cls_1):
            cls_0 = cls_0.append(cls_0)
        
        return cls_1.append(cls_0.iloc[:len(cls_1)])

# READ DATA

# data = read_ensemble_structured()
# X, y = data.iloc[:, :-2], data.iloc[:, -2]

# first_step = {}

# PRE-PROCESSING TRAIN

# clf_1 = joblib.load('out/gb_3.pkl')
# first_step['gb'] = clf_1.predict(X)

# clf_2 = joblib.load('out/rf_2.pkl')
# first_step['rf'] = clf_2.predict(X)

# clf_3 = joblib.load('out/ab.pkl')
# first_step['ab'] = clf_3.predict(X)

# clf_4 = load_model('out/mlp.h5')
# first_step['mlp'] = clf_4.predict_classes(X).flatten()

# clf_5 = load('out/linear/lr.joblib')
# first_step['lr'] = clf_5.predict(X)

# values = [
#     first_step['gb'],
#     first_step['rf'],
#     first_step['ab'],
#     first_step['mlp'],
#     first_step['lr'],
# ]

# first_step['sum'] = [float(sum(x)) / float(len(x)) for x in zip(*values)]
# first_step['value'] = y

# pd.DataFrame(first_step).to_csv('data/ensemble_train_parsed.csv', sep='\t')

# PRE-PROCESSING TEST

# data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
# X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]

# first_step = {}

# first_step['gb'] = clf_1.predict(X_test)
# first_step['rf'] = clf_2.predict(X_test)
# first_step['ab'] = clf_3.predict(X_test)
# first_step['mlp'] = clf_4.predict_classes(X_test).flatten()
# first_step['lr'] = clf_5.predict(X_test)

# values = [
#     first_step['gb'],
#     first_step['rf'],
#     first_step['ab'],
#     first_step['mlp'],
#     first_step['lr'],
# ]

# first_step['sum'] = [float(sum(x)) / float(len(x)) for x in zip(*values)]
# first_step['value'] = y_test

# pd.DataFrame(first_step).to_csv('data/test_parsed.csv', sep='\t')

# DEBUG

# project = Project()
# project.display_metrics(clf_1, X_test, y_test, mode=1)

# project = Project()
# project.display_metrics(clf_2, X_test, y_test, mode=1)

# project = Project()
# project.display_metrics(clf_3, X_test, y_test, mode=1)

# project = Project()
# project.display_metrics(clf_4, X_test, y_test)

# project = Project()
# project.display_metrics(clf_5, X_test, y_test, mode=1)

# ENSEMBLE

data_ensemble = pd.read_csv('data/ensemble_train_parsed.csv', sep='\t', header=0, index_col=0)
data_ensemble = balance_ensemble_data(data_ensemble)
data_ensemble = shuffle(data_ensemble)

X_train, y_train = data_ensemble.iloc[:, :-1], data_ensemble.iloc[:, -1]

data_test = pd.read_csv('data/test_parsed.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-1], data_test.iloc[:, -1]

# ENSEMBLE (MLP)

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=6))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, callbacks=[EarlyStopping(patience=50)], epochs=100000, batch_size=64, validation_split=0.3, verbose=1)
model.save('out/final/mlp_3.h5')

project = Project()
project.display_metrics(model, X_test, y_test, history, display_losses=True)

# ENSEMBLE (NB)

clf = GaussianNB()
clf.fit(X_train, y_train)

with open('out/final/nb_3.pkl', 'wb') as f:
    pickle.dump(clf, f)

project = Project()
project.display_metrics(clf, X_test, y_test, mode=1)

# ENSEMBLE (DT)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

with open('out/final/dt_3.pkl', 'wb') as f:
    pickle.dump(clf, f)

project = Project()
project.display_metrics(clf, X_test, y_test, mode=1)