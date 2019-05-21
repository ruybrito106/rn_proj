import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle

import pickle
from sklearn.externals import joblib

from project import Project

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]
    while len(cls_1) < len(cls_0):
        cls_1 = cls_1.append(cls_1)
        
    return cls_0.append(cls_1.iloc[:len(cls_0)])

# READ DATA

data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=200)
data = balance_data(data)
data = shuffle(data)

X, y = data.iloc[:, :-2], data.iloc[:, -2]
X = SelectKBest(k=30).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)

# TRAIN

clf = GaussianNB()
history = clf.fit(X_train, y_train)

with open('out/nb.pkl', 'wb') as f:
    pickle.dump(clf, f)

# CLASSIFY

clf2 = joblib.load('out/nb.pkl')
project = Project()
project.display_metrics(clf2, X, y, mode=1)