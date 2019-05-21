import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_curve, auc

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]

    while len(cls_1) < len(cls_0):
        cls_1 = cls_1.append(cls_1)

    return cls_0.append(cls_1.iloc[:len(cls_0)])

def get_roc_auc(y, pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, pred)
    return auc(false_positive_rate, true_positive_rate)

# READ DATA

data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=100)
data = balance_data(data)

X, y = data.iloc[:, 1:-2], data.iloc[:, -2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# LEARNING RATE TUNNING

options = [0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1]

lr_train = []
lr_test = []

for x in options:
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=x)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    lr_train.append(get_roc_auc(y_train, train_pred))

    test_pred = clf.predict(X_test)
    lr_test.append(get_roc_auc(y_test, test_pred))

    print (x)

print (lr_train)
print (lr_test)

# ESTIMATORS TUNNING

options = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 750, 1024]

est_train = []
est_test = []

for x in options:
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    est_train.append(get_roc_auc(y_train, train_pred))

    test_pred = clf.predict(X_test)
    est_test.append(get_roc_auc(y_test, test_pred))

    print (x)

print (est_train)
print (est_test)