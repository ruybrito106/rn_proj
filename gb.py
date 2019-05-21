import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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

data = pd.read_csv('data/test.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=200)
data = balance_data(data)
data = shuffle(data)

X, y = data.iloc[:, 1:-2], data.iloc[:, -2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)

# TRAIN

# clf = GradientBoostingClassifier(n_estimators=512, max_features=176, max_depth=11, learning_rate=0.2, verbose=1)
# history = clf.fit(X_train, y_train)

# with open('out/gb_3.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# CLASSIFY

clf2 = joblib.load('out/gb_3.pkl')
project = Project()
project.display_metrics(clf2, X, y, mode=1)