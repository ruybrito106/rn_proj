import pandas as pd

from joblib import dump, load
from sklearn.externals import joblib
from sklearn.metrics import roc_curve

from matplotlib import pyplot

from keras.models import load_model

# GET CLASSIFIERS

clfs = list()

clfs.append(joblib.load('out/gb_3.pkl'))
clfs.append(joblib.load('out/rf_2.pkl'))
clfs.append(joblib.load('out/ab.pkl'))
clfs.append(load_model('out/mlp.h5'))
clfs.append(load('out/linear/lr.joblib'))

clfs.append(load_model('out/final/mlp_3.h5'))
clfs.append(joblib.load('out/final/nb_3.pkl'))
clfs.append(joblib.load('out/final/dt_3.pkl'))

# READ TEST DATA

data_test = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
X_test, y_test = data_test.iloc[:, :-2], data_test.iloc[:, -2]

data_test_parsed = pd.read_csv('data/test_parsed.csv', sep='\t', header=0, index_col=0)
_X_test, _y_test = data_test_parsed.iloc[:, :-1], data_test_parsed.iloc[:, -1]

# PLOT ROC CURVES

NAMES = [
    'Gradient Boosting Classifier', 
    'Random Forest Classifier', 
    'AdaBoost Classifier', 
    'MLP Classifier', 
    'Logistic Regression Classifier', 
    'MLP Metaclassifier', 
    'Naive Bayes Metaclassifier', 
    'Decision Tree Metaclassifier',
]

COLORS = [
    'cyan',
    'green',
    'green',
    'green',
    'green',
    'red',
    'green',
    'yellow',
]

def plot_curve(labels, probs, index):
    if index != 3 and index != 5:
        probs = map(lambda x : x[1], probs)

    fpr, tpr, _ = roc_curve(labels, probs)
    pyplot.plot(fpr, tpr, color=COLORS[index], label=NAMES[index])

pyplot.plot([0, 1], [0, 1], linestyle='--', label='Divisor')

for i in range(len(clfs)):
    if i >= 1 and i <= 4:
        continue

    if i < 5:
        plot_curve(y_test, clfs[i].predict_proba(X_test), i)
    else:
        plot_curve(_y_test, clfs[i].predict_proba(_X_test), i)

pyplot.gca().legend(('Divisor', 'Gradient Boosting Classifier', 'MLP Metaclassifier', 'Naive Bayes Metaclassifier', 'Decision Tree Metaclassifier'))
pyplot.show()