import pandas as pd

from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle

import pickle
from sklearn.externals import joblib

from keep_best import GetBest
from project import Project
from structure_experiment import read_chunk_structured

# READ DATA

# raw = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
# X, y = raw.iloc[:, :-2], raw.iloc[:, -2]

# selection = SelectKBest(k=180)
# selection.fit_transform(X, y)
# fs = selection.get_support()

data = read_chunk_structured(index=3)
X_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
# X_train = X_train.iloc[:, fs]

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]
# X_test = X_test.iloc[:, fs]

# CREATE MODEL

# def init_model_fn():
#     model = Sequential()
    # model.add(Dense(180, activation='relu', input_dim=180))
#     model.add(Dense(64, activation='softplus'))
#     model.add(Dense(1))

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#     return model

# CLASSIFY

# clf = init_model_fn()
# history = clf.fit(X_train, y_train, callbacks=[EarlyStopping(patience=100)], epochs=10000, batch_size=256, validation_split=0.3)
# clf.save('out/mlp_5.h5')

clf2 = load_model('out/mlp.h5')
for i in range(3):
    layer = clf2.get_layer(index=i)
    print (layer.get_config())

print (clf2.summary())

project = Project()
# project.display_metrics(clf2, X_test, y_test, history)